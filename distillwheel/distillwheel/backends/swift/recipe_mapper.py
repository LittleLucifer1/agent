"""Translate a DistillWheel recipe into an ms-swift 4.4 YAML config.

The generated file is passed positionally to ``swift sft`` or
``swift rlhf``.  It therefore contains only ms-swift arguments; launcher-only
sentinels must never be written into it.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict

from ...core.errors import IRValidationError
from ...core.ir.recipe import Recipe

# Preference methods share ms-swift's ``rlhf`` command and select the
# algorithm with ``rlhf_type``.
_SUBCOMMAND = {
    "sft": ("sft", None),
    "dpo": ("rlhf", "dpo"),
    "kto": ("rlhf", "kto"),
}

# DistillWheel owns these resources.  Letting an escape-hatch override replace
# them would make the normalized recipe disagree with the actual run.
_PROTECTED_OVERRIDE_KEYS = {
    "__subcommand__",
    "dataset",
    "val_dataset",
    "output_dir",
}


def swift_subcommand_for(stage: str) -> tuple[str, str | None]:
    if stage not in _SUBCOMMAND:
        raise IRValidationError(f"swift adapter does not support stage={stage!r}")
    return _SUBCOMMAND[stage]


def _peft_block(recipe: Recipe) -> Dict[str, Any]:
    peft = recipe.peft
    if peft is None or peft.type == "full":
        return {"tuner_type": "full"}
    if peft.type not in {"lora", "qlora"}:
        raise IRValidationError(f"unsupported swift PEFT type: {peft.type!r}")

    # ms-swift 4.x represents QLoRA as LoRA over a bitsandbytes-quantized base
    # model; ``qlora`` is not itself a valid tuner_type.
    block: Dict[str, Any] = {
        "tuner_type": "lora",
        "lora_rank": peft.r,
        "lora_alpha": peft.alpha,
        "lora_dropout": peft.dropout,
    }
    if peft.target_modules:
        block["target_modules"] = list(peft.target_modules)
    if peft.type == "qlora":
        block.update({"quant_method": "bnb", "quant_bits": 4})
    return block


def _torch_dtype(precision: str) -> str:
    if precision == "bf16":
        return "bfloat16"
    if precision == "fp16":
        return "float16"
    if precision == "fp8":
        raise IRValidationError(
            "ms-swift 4.4 does not accept recipe precision='fp8' as a "
            "torch_dtype; use bf16/fp16 or configure a supported FP8 workflow "
            "outside this adapter"
        )
    raise IRValidationError(f"unsupported swift precision: {precision!r}")


def _swift_overrides(recipe: Recipe) -> Dict[str, Any]:
    swift_meta = (recipe.meta or {}).get("swift")
    if swift_meta is None:
        return {}
    if not isinstance(swift_meta, Mapping):
        raise IRValidationError("recipe.meta.swift must be a mapping")

    overrides = swift_meta.get("overrides")
    if overrides is None:
        return {}
    if not isinstance(overrides, Mapping):
        raise IRValidationError("recipe.meta.swift.overrides must be a mapping")

    result: Dict[str, Any] = {}
    for key, value in overrides.items():
        if not isinstance(key, str) or not key or key.startswith("-"):
            raise IRValidationError("swift override keys must be non-empty argument names")
        if key in _PROTECTED_OVERRIDE_KEYS:
            raise IRValidationError(
                f"recipe.meta.swift.overrides cannot replace DistillWheel-owned "
                f"key {key!r}"
            )
        result[key] = value
    return result


def recipe_to_swift_args(recipe: Recipe, data_path: Path, cfg_path: Path) -> Path:
    """Write an ms-swift 4.4 YAML config and return its absolute path."""
    import yaml

    if recipe.parallel.tp > 1 or recipe.parallel.pp > 1:
        raise IRValidationError(
            "the swift transformers launcher supports DDP/DeepSpeed only; "
            "parallel.tp and parallel.pp must remain 1 (use a future "
            "Megatron-SWIFT adapter for tensor/pipeline parallelism)"
        )

    _, rlhf_type = swift_subcommand_for(recipe.stage)
    cfg_path = Path(cfg_path).expanduser().resolve()
    data_path = Path(data_path).expanduser().resolve()
    native_output = (cfg_path.parent / "swift_native").resolve()

    cfg: Dict[str, Any] = {
        "model": recipe.base_model,
        "dataset": [str(data_path)],
        "output_dir": str(native_output),
        "num_train_epochs": recipe.train.epochs,
        "per_device_train_batch_size": recipe.train.micro_batch,
        "gradient_accumulation_steps": recipe.train.grad_accum,
        "learning_rate": recipe.optim.lr,
        "lr_scheduler_type": recipe.optim.scheduler,
        "warmup_ratio": recipe.optim.warmup_ratio,
        "weight_decay": recipe.optim.weight_decay,
        "max_length": recipe.train.max_len,
        "seed": recipe.train.seed,
        "save_strategy": "steps",
        "save_steps": recipe.io.save_steps,
        "logging_steps": recipe.io.logging_steps,
        # DistillWheel currently supplies no validation stream.  Disabling the
        # automatic split also keeps tiny smoke datasets usable.
        "split_dataset_ratio": 0.0,
        # Keep the native output stable at <workdir>/swift_native.  The
        # checkpoint normalizer remains defensive about legacy version dirs.
        "add_version": False,
        "torch_dtype": _torch_dtype(recipe.precision),
    }
    cfg.update(_peft_block(recipe))

    if recipe.target_template:
        cfg["template"] = recipe.target_template

    if rlhf_type is not None:
        cfg["rlhf_type"] = rlhf_type

    if recipe.io.resume_from:
        resume_path = Path(recipe.io.resume_from).expanduser().resolve()
        cfg["resume_from_checkpoint"] = str(resume_path)

    # Preflight hook: cap to a single step if the orchestrator asked.
    if recipe.meta.get("__preflight__"):
        cfg["max_steps"] = int(recipe.meta.get("max_steps", 1))

    zero_stage = recipe.parallel.zero_stage
    if zero_stage not in {0, 1, 2, 3}:
        raise IRValidationError(
            "ms-swift's built-in DeepSpeed configs support zero_stage 0, 1, 2, or 3"
        )
    if zero_stage:
        cfg["deepspeed"] = f"zero{zero_stage}"

    # This is deliberately last so advanced ms-swift 4.4 arguments can be
    # supplied without expanding the Recipe schema.  Owned paths/data above
    # cannot be replaced.
    cfg.update(_swift_overrides(recipe))

    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return cfg_path
