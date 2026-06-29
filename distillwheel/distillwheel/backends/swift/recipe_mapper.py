"""Recipe → swift YAML config translator.

We emit a minimal config that ms-swift's ``swift sft / swift rlhf``
commands accept. Field names follow swift's conventions
(``--num_train_epochs``, ``--per_device_train_batch_size``...) which
swift's YAML loader maps onto its argparse spec.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ...core.ir.recipe import Recipe

# Stage ↔ swift sub-command. swift uses `sft` for SFT, `rlhf` for the
# preference-style methods (DPO/KTO) with a `--rlhf_type` switch.
_SUBCOMMAND = {
    "sft": ("sft", None),
    "dpo": ("rlhf", "dpo"),
    "kto": ("rlhf", "kto"),
}


def swift_subcommand_for(stage: str) -> tuple[str, str | None]:
    if stage not in _SUBCOMMAND:
        raise ValueError(f"swift adapter does not support stage={stage!r}")
    return _SUBCOMMAND[stage]


def _peft_block(recipe: Recipe) -> Dict[str, Any]:
    if recipe.peft is None or recipe.peft.type == "full":
        return {"sft_type": "full"}
    block = {
        "sft_type": "lora" if recipe.peft.type == "lora" else "qlora",
        "lora_rank": recipe.peft.r,
        "lora_alpha": recipe.peft.alpha,
        "lora_dropout": recipe.peft.dropout,
    }
    if recipe.peft.target_modules:
        block["lora_target_modules"] = list(recipe.peft.target_modules)
    return block


def recipe_to_swift_args(recipe: Recipe, data_path: Path, cfg_path: Path) -> Path:
    """Write a swift-style YAML config and return its path."""
    import yaml

    cmd, rlhf_type = swift_subcommand_for(recipe.stage)

    cfg: Dict[str, Any] = {
        "model": recipe.base_model,
        "dataset": [str(data_path)],
        "output_dir": str(Path(recipe.io.output_dir) / "workdir" / "swift_native"),
        "num_train_epochs": recipe.train.epochs,
        "per_device_train_batch_size": recipe.train.micro_batch,
        "gradient_accumulation_steps": recipe.train.grad_accum,
        "learning_rate": recipe.optim.lr,
        "lr_scheduler_type": recipe.optim.scheduler,
        "warmup_ratio": recipe.optim.warmup_ratio,
        "weight_decay": recipe.optim.weight_decay,
        "max_length": recipe.train.max_len,
        "seed": recipe.train.seed,
        "save_steps": recipe.io.save_steps,
        "logging_steps": recipe.io.logging_steps,
        "bf16": recipe.precision == "bf16",
        "fp16": recipe.precision == "fp16",
    }
    cfg.update(_peft_block(recipe))

    if recipe.target_template:
        cfg["template"] = recipe.target_template

    if rlhf_type is not None:
        cfg["rlhf_type"] = rlhf_type

    if recipe.io.resume_from:
        cfg["resume_from_checkpoint"] = recipe.io.resume_from

    # Preflight hook: cap to a single step if the orchestrator asked.
    if recipe.meta.get("__preflight__"):
        cfg["max_steps"] = int(recipe.meta.get("max_steps", 1))

    # ZeRO stage if any
    if recipe.parallel.zero_stage:
        cfg["deepspeed"] = f"zero{recipe.parallel.zero_stage}"

    cfg["__subcommand__"] = cmd  # consumed by the launcher; stripped before swift sees it

    cfg_path = Path(cfg_path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return cfg_path
