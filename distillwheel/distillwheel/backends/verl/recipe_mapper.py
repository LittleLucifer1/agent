"""Translate DistillWheel recipes into VERL 0.8 Hydra overrides."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ...core.errors import IRValidationError
from ...core.ir.recipe import Recipe

# VERL selects all three algorithms through ``main_ppo``.  The distinguishing
# setting is the advantage estimator, not a non-existent top-level ``algorithm``
# scalar override.
_ADV_ESTIMATOR = {"grpo": "grpo", "ppo": "gae", "rloo": "rloo"}
_CUSTOM_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
_MODULE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_FUNCTION = re.compile(r"^[A-Za-z_]\w*$")
_CUSTOM_PREFIXES = (
    "actor_rollout_ref.", "critic.", "algorithm.", "data.", "trainer.", "reward.",
)
_PROTECTED_CUSTOM_KEYS = {
    "actor_rollout_ref.model.path",
    "critic.model.path",
    "data.train_files",
    "data.val_files",
    "trainer.default_local_dir",
    "reward.custom_reward_function.path",
    "reward.custom_reward_function.name",
}


def verl_algorithm_for(stage: str) -> str:
    """Return VERL's advantage-estimator name for a DistillWheel stage."""
    try:
        return _ADV_ESTIMATOR[stage]
    except KeyError as exc:
        raise IRValidationError(
            f"verl adapter supports only grpo, ppo and rloo; got stage={stage!r}"
        ) from exc


def _positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise IRValidationError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise IRValidationError(f"{name} must be > 0, got {value!r}")
    return float(value)


def _number_in_range(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IRValidationError(f"{name} must be numeric, got {value!r}")
    value = float(value)
    if not low < value < high:
        raise IRValidationError(f"{name} must be in ({low}, {high}), got {value!r}")
    return value


def _hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        # Keep familiar model identifiers readable.  JSON quoting handles
        # Windows paths, whitespace and multi-line Jinja templates safely.
        if re.fullmatch(r"[A-Za-z0-9_./@+%:-]+", value) and "\\" not in value:
            return value
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    raise IRValidationError(f"unsupported Hydra override value type: {type(value).__name__}")


def _reward_function(ref: str, workdir: Path) -> Tuple[Path, str]:
    if not isinstance(ref, str) or ":" not in ref:
        raise IRValidationError(
            "recipe.rl.reward_fn_ref must be '/absolute/file.py:function' or "
            "'dotted.module:function'"
        )
    source, function = (part.strip() for part in ref.rsplit(":", 1))
    if not source or not _FUNCTION.fullmatch(function):
        raise IRValidationError(f"invalid reward function reference: {ref!r}")

    candidate = Path(source).expanduser()
    is_file_ref = candidate.is_absolute() or source.endswith(".py") or "/" in source or "\\" in source
    if is_file_ref:
        if candidate.suffix.lower() != ".py":
            raise IRValidationError(f"reward function file must end in .py: {source!r}")
        try:
            # Relative references are resolved on the configuration-generating
            # machine.  Hydra always receives the resulting absolute path so
            # Ray workers never depend on their own cwd.
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise IRValidationError(f"reward function file does not exist: {candidate}") from exc
        if not resolved.is_file():
            raise IRValidationError(f"reward function path is not a file: {resolved}")
        return resolved, function

    if not _MODULE.fullmatch(source):
        raise IRValidationError(f"invalid dotted reward module: {source!r}")

    # VERL 0.8 requires a Python *file*, whereas Recipe deliberately permits a
    # dotted import reference.  A tiny workdir-local shim bridges the two and
    # is visible to every worker when the workdir is on shared storage.
    workdir.mkdir(parents=True, exist_ok=True)
    shim = (workdir / "distillwheel_verl_reward.py").resolve()
    shim.write_text(
        "\"\"\"Generated by DistillWheel; do not edit.\"\"\"\n"
        "from importlib import import_module as _import_module\n\n"
        f"{function} = getattr(_import_module({source!r}), {function!r})\n",
        encoding="utf-8",
    )
    return shim, function


def _validate_recipe(recipe: Recipe) -> Mapping[str, Any]:
    if recipe.rl is None:
        raise IRValidationError("verl adapter requires recipe.rl")
    verl_algorithm_for(recipe.stage)

    if recipe.precision == "fp8":
        raise IRValidationError("verl 0.8 backend does not support recipe precision='fp8'")
    if recipe.precision not in ("bf16", "fp16"):
        raise IRValidationError(f"unsupported verl precision: {recipe.precision!r}")
    if recipe.peft is not None and recipe.peft.type == "qlora":
        raise IRValidationError("verl 0.8 backend does not support QLoRA")
    if recipe.peft is not None and recipe.peft.type not in ("full", "lora"):
        raise IRValidationError(f"unsupported verl PEFT type: {recipe.peft.type!r}")
    if (
        recipe.peft is not None
        and recipe.peft.type == "lora"
        and recipe.peft.dropout != 0
    ):
        raise IRValidationError(
            "VERL 0.8's FSDP LoRA path does not pass dropout to PEFT; "
            "recipe.peft.dropout must be 0"
        )
    if recipe.parallel.zero_stage != 0:
        raise IRValidationError(
            "recipe.parallel.zero_stage is an ms-swift/DeepSpeed setting and cannot be mapped to VERL 0.8"
        )
    if recipe.parallel.pp != 1:
        raise IRValidationError("verl 0.8 backend currently requires recipe.parallel.pp=1")

    dp = _positive_int(recipe.parallel.dp, "recipe.parallel.dp")
    tp = _positive_int(recipe.parallel.tp, "recipe.parallel.tp")
    if dp % tp:
        raise IRValidationError("recipe.parallel.tp must divide recipe.parallel.dp for VERL rollout")
    _positive_int(recipe.train.global_batch, "recipe.train.global_batch")
    _positive_int(recipe.train.micro_batch, "recipe.train.micro_batch")
    if recipe.train.global_batch < recipe.train.micro_batch:
        raise IRValidationError("recipe.train.global_batch must be >= recipe.train.micro_batch")
    _positive_int(recipe.train.max_len, "recipe.train.max_len")
    if isinstance(recipe.train.epochs, bool) or not isinstance(recipe.train.epochs, (int, float)):
        raise IRValidationError("VERL epochs must be a positive whole number")
    if recipe.train.epochs <= 0 or not float(recipe.train.epochs).is_integer():
        raise IRValidationError("VERL epochs must be a positive whole number")
    _positive_number(recipe.optim.lr, "recipe.optim.lr")
    if not 0 <= recipe.optim.warmup_ratio < 1:
        raise IRValidationError("recipe.optim.warmup_ratio must be in [0, 1)")
    if recipe.optim.weight_decay < 0:
        raise IRValidationError("recipe.optim.weight_decay must be >= 0")
    if recipe.optim.scheduler not in ("constant", "cosine"):
        raise IRValidationError("VERL 0.8 optimizer scheduler must be 'constant' or 'cosine'")
    if recipe.io.save_steps == 0 or recipe.io.save_steps < -1:
        raise IRValidationError("recipe.io.save_steps must be -1 or a positive integer")

    rollout_n = _positive_int(recipe.rl.rollout_n, "recipe.rl.rollout_n")
    if recipe.stage in ("grpo", "rloo") and rollout_n < 2:
        raise IRValidationError(f"{recipe.stage} requires recipe.rl.rollout_n >= 2")
    if recipe.rl.rollout_engine not in ("vllm", "sglang"):
        raise IRValidationError(f"unsupported VERL rollout engine: {recipe.rl.rollout_engine!r}")
    if recipe.rl.kl_coef < 0:
        raise IRValidationError("recipe.rl.kl_coef must be >= 0")
    if not 0 < recipe.rl.clip <= 1:
        raise IRValidationError("recipe.rl.clip must be in (0, 1]")

    if not isinstance(recipe.meta, Mapping):
        raise IRValidationError("recipe.meta must be a mapping")
    meta = recipe.meta.get("verl", {}) or {}
    if not isinstance(meta, Mapping):
        raise IRValidationError("recipe.meta.verl must be a mapping")
    return meta


def _set_custom_overrides(config: Dict[str, str], custom: Any) -> None:
    if custom is None:
        return
    items: List[Tuple[str, str]] = []
    if isinstance(custom, Mapping):
        items = [(str(key), _hydra_value(value)) for key, value in custom.items()]
    elif isinstance(custom, Sequence) and not isinstance(custom, (str, bytes)):
        for item in custom:
            if not isinstance(item, str) or "=" not in item or "\n" in item or "\r" in item:
                raise IRValidationError(
                    "recipe.meta.verl.overrides list entries must be one-line 'key=value' strings"
                )
            key, value = item.split("=", 1)
            items.append((key.strip(), value))
    else:
        raise IRValidationError("recipe.meta.verl.overrides must be a mapping or a list")

    for key, value in items:
        if not _CUSTOM_KEY.fullmatch(key) or not key.startswith(_CUSTOM_PREFIXES):
            raise IRValidationError(f"forbidden or malformed VERL override key: {key!r}")
        if key in _PROTECTED_CUSTOM_KEYS:
            raise IRValidationError(f"VERL override key is managed by DistillWheel: {key}")
        config[key] = value  # Deliberately last-wins, but every key remains unique.


def recipe_to_verl_overrides(recipe: Recipe, data_path: Path, out_path: Path) -> Path:
    meta = _validate_recipe(recipe)
    assert recipe.rl is not None

    data_path = Path(data_path).expanduser().resolve()
    if not data_path.is_file():
        raise IRValidationError(f"VERL parquet file does not exist: {data_path}")
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    max_response_length = _positive_int(
        meta.get("max_response_length", 256), "recipe.meta.verl.max_response_length"
    )
    gpu_memory = _number_in_range(
        meta.get("gpu_memory_utilization", 0.4),
        "recipe.meta.verl.gpu_memory_utilization", 0.0, 1.0,
    )
    actor_mini = _positive_int(
        meta.get("ppo_mini_batch_size", recipe.train.global_batch),
        "recipe.meta.verl.ppo_mini_batch_size",
    )
    actor_micro = _positive_int(
        meta.get("ppo_micro_batch_size_per_gpu", recipe.train.micro_batch),
        "recipe.meta.verl.ppo_micro_batch_size_per_gpu",
    )
    rollout_logprob_micro = _positive_int(
        meta.get("rollout_log_prob_micro_batch_size_per_gpu", recipe.train.micro_batch),
        "recipe.meta.verl.rollout_log_prob_micro_batch_size_per_gpu",
    )
    ref_logprob_micro = _positive_int(
        meta.get("ref_log_prob_micro_batch_size_per_gpu", recipe.train.micro_batch),
        "recipe.meta.verl.ref_log_prob_micro_batch_size_per_gpu",
    )

    dtype = {"bf16": "bfloat16", "fp16": "float16"}[recipe.precision]
    native_dir = (out_path.parent / "verl_native").resolve()
    config: Dict[str, str] = {}

    def put(key: str, value: Any) -> None:
        config[key] = _hydra_value(value)

    # Data: validation is intentionally disabled, but VERL still resolves the
    # val_files field while composing its config, so point it at the same valid
    # parquet instead of creating a dummy or leaving the example default.
    put("data.train_files", data_path)
    put("data.val_files", data_path)
    put("data.train_batch_size", recipe.train.global_batch)
    put("data.max_prompt_length", recipe.train.max_len)
    put("data.max_response_length", max_response_length)
    put("data.seed", recipe.train.seed)
    put("data.truncation", "error")

    put("algorithm.adv_estimator", verl_algorithm_for(recipe.stage))
    put("actor_rollout_ref.model.path", recipe.base_model)
    put("actor_rollout_ref.rollout.name", recipe.rl.rollout_engine)
    put("actor_rollout_ref.rollout.mode", meta.get("rollout_mode", "sync"))
    put("actor_rollout_ref.rollout.n", recipe.rl.rollout_n)
    put("actor_rollout_ref.rollout.dtype", dtype)
    put("actor_rollout_ref.rollout.tensor_model_parallel_size", recipe.parallel.tp)
    put("actor_rollout_ref.rollout.gpu_memory_utilization", gpu_memory)
    put("actor_rollout_ref.rollout.enforce_eager", meta.get("enforce_eager", True))
    put("actor_rollout_ref.rollout.free_cache_engine", meta.get("free_cache_engine", True))
    put("actor_rollout_ref.rollout.enable_chunked_prefill", meta.get("enable_chunked_prefill", False))
    put("actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu", rollout_logprob_micro)
    put("actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu", ref_logprob_micro)
    for key in ("max_num_batched_tokens", "max_model_len", "max_num_seqs", "temperature", "top_p", "top_k"):
        if key in meta:
            put(f"actor_rollout_ref.rollout.{key}", meta[key])

    put("actor_rollout_ref.actor.ppo_mini_batch_size", actor_mini)
    put("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", actor_micro)
    put("actor_rollout_ref.actor.clip_ratio", recipe.rl.clip)
    put("actor_rollout_ref.actor.optim.lr", recipe.optim.lr)
    put("actor_rollout_ref.actor.optim.lr_warmup_steps_ratio", recipe.optim.warmup_ratio)
    put("actor_rollout_ref.actor.optim.weight_decay", recipe.optim.weight_decay)
    put("actor_rollout_ref.actor.optim.lr_scheduler_type", recipe.optim.scheduler)
    put("actor_rollout_ref.actor.data_loader_seed", recipe.train.seed)
    put("actor_rollout_ref.actor.fsdp_config.dtype", dtype)
    put("actor_rollout_ref.actor.fsdp_config.seed", recipe.train.seed)
    put("actor_rollout_ref.ref.fsdp_config.dtype", dtype)
    put("actor_rollout_ref.ref.fsdp_config.seed", recipe.train.seed)

    # VERL recommends actor KL loss for critic-free estimators, while PPO's
    # GAE path uses the in-reward KL controller.
    use_kl = recipe.rl.kl_coef > 0
    if recipe.stage == "ppo":
        put("actor_rollout_ref.actor.use_kl_loss", False)
        put("algorithm.use_kl_in_reward", use_kl)
        put("algorithm.kl_ctrl.kl_coef", recipe.rl.kl_coef)
    else:
        put("actor_rollout_ref.actor.use_kl_loss", use_kl)
        put("actor_rollout_ref.actor.kl_loss_coef", recipe.rl.kl_coef)
        put("actor_rollout_ref.actor.kl_loss_type", meta.get("kl_loss_type", "low_var_kl"))
        put("algorithm.use_kl_in_reward", False)

    put("critic.enable", recipe.stage == "ppo")
    if recipe.stage == "ppo":
        critic_mini = _positive_int(
            meta.get("critic_ppo_mini_batch_size", actor_mini),
            "recipe.meta.verl.critic_ppo_mini_batch_size",
        )
        critic_micro = _positive_int(
            meta.get("critic_ppo_micro_batch_size_per_gpu", actor_micro),
            "recipe.meta.verl.critic_ppo_micro_batch_size_per_gpu",
        )
        critic_lr = _positive_number(
            meta.get("critic_lr", recipe.optim.lr), "recipe.meta.verl.critic_lr"
        )
        put("critic.model.path", meta.get("critic_model", recipe.base_model))
        put("critic.ppo_mini_batch_size", critic_mini)
        put("critic.ppo_micro_batch_size_per_gpu", critic_micro)
        put("critic.optim.lr", critic_lr)
        put("critic.optim.lr_warmup_steps_ratio", recipe.optim.warmup_ratio)
        put("critic.optim.weight_decay", recipe.optim.weight_decay)
        put("critic.optim.lr_scheduler_type", recipe.optim.scheduler)
        put("critic.data_loader_seed", recipe.train.seed)
        put("critic.fsdp.dtype", dtype)
        put("critic.fsdp.seed", recipe.train.seed)

    if recipe.peft is not None and recipe.peft.type == "lora":
        _positive_int(recipe.peft.r, "recipe.peft.r")
        _positive_int(recipe.peft.alpha, "recipe.peft.alpha")
        if not 0 <= recipe.peft.dropout < 1:
            raise IRValidationError("recipe.peft.dropout must be in [0, 1)")
        # These top-level fields are the VERL 0.8 FSDP LoRA interface.  The
        # nested ``model.lora`` block is for Megatron and must not be used to
        # pretend FSDP supports a dropout value that it silently ignores.
        put("actor_rollout_ref.model.lora_rank", recipe.peft.r)
        put("actor_rollout_ref.model.lora_alpha", recipe.peft.alpha)
        put("actor_rollout_ref.model.target_modules", recipe.peft.target_modules or "all-linear")

    custom_template = meta.get("custom_chat_template")
    if custom_template is not None:
        if not isinstance(custom_template, str) or not custom_template.strip():
            raise IRValidationError("recipe.meta.verl.custom_chat_template must be a non-empty Jinja string")
        if "{{" not in custom_template and "{%" not in custom_template:
            raise IRValidationError(
                "recipe.meta.verl.custom_chat_template must contain Jinja expressions, not a template name"
            )
        put("actor_rollout_ref.model.custom_chat_template", custom_template)
        if recipe.stage == "ppo":
            put("critic.model.custom_chat_template", custom_template)

    if recipe.rl.reward_fn_ref:
        reward_path, reward_name = _reward_function(recipe.rl.reward_fn_ref, out_path.parent)
        put("reward.custom_reward_function.path", reward_path)
        put("reward.custom_reward_function.name", reward_name)

    put("trainer.total_epochs", int(recipe.train.epochs))
    put("trainer.default_local_dir", native_dir)
    put("trainer.save_freq", recipe.io.save_steps)
    put("trainer.logger", ["console"])
    put("trainer.val_before_train", False)
    put("trainer.test_freq", -1)
    put("trainer.n_gpus_per_node", recipe.parallel.dp)
    put("trainer.nnodes", 1)
    if recipe.io.resume_from:
        put("trainer.resume_mode", "resume_path")
        put("trainer.resume_from_path", Path(recipe.io.resume_from).expanduser().resolve())
    else:
        put("trainer.resume_mode", "disable")

    if recipe.meta.get("__preflight__"):
        max_steps = _positive_int(recipe.meta.get("max_steps", 1), "recipe.meta.max_steps")
        put("trainer.total_training_steps", max_steps)

    _set_custom_overrides(config, meta.get("overrides"))
    out_path.write_text(
        "".join(f"{key}={value}\n" for key, value in config.items()),
        encoding="utf-8",
    )
    return out_path


def load_overrides(path: Path) -> List[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.strip() and not line.lstrip().startswith("#")]
