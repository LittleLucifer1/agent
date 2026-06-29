"""Recipe → verl hydra-override translator.

verl uses Hydra: the trainer takes a base config (e.g.
``trainer/ppo``) plus a list of ``key=value`` overrides. We emit those
overrides to a text file, one per line, so the launcher can ``cat`` them
into the argv (Hydra accepts repeated ``+key=value`` args).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from ...core.errors import IRValidationError
from ...core.ir.recipe import Recipe

# stage → verl algorithm name
_ALGO = {
    "grpo": "grpo",
    "ppo": "ppo",
    "rloo": "rloo",
    "opd": "opd",
}


def verl_algorithm_for(stage: str) -> str:
    if stage not in _ALGO:
        raise IRValidationError(f"verl adapter does not support stage={stage!r}")
    return _ALGO[stage]


def recipe_to_verl_overrides(recipe: Recipe, data_path: Path, out_path: Path) -> Path:
    if recipe.rl is None:
        raise IRValidationError("verl adapter requires recipe.rl")

    algo = verl_algorithm_for(recipe.stage)

    overrides: List[str] = [
        f"algorithm={algo}",
        f"actor_rollout_ref.model.path={recipe.base_model}",
        f"actor_rollout_ref.rollout.name={recipe.rl.rollout_engine}",
        f"actor_rollout_ref.rollout.n={recipe.rl.rollout_n}",
        f"actor_rollout_ref.actor.optim.lr={recipe.optim.lr}",
        f"actor_rollout_ref.actor.kl_loss_coef={recipe.rl.kl_coef}",
        f"actor_rollout_ref.actor.clip_ratio={recipe.rl.clip}",
        f"data.train_files={data_path}",
        f"data.train_batch_size={recipe.train.global_batch}",
        f"data.max_prompt_length={recipe.train.max_len}",
        f"trainer.total_epochs={recipe.train.epochs}",
        f"trainer.default_local_dir={Path(recipe.io.output_dir).resolve() / 'workdir' / 'verl_native'}",
        f"trainer.save_freq={recipe.io.save_steps}",
        f"trainer.logger.console.log_freq={recipe.io.logging_steps}",
        f"trainer.seed={recipe.train.seed}",
    ]

    # PEFT
    if recipe.peft is not None and recipe.peft.type != "full":
        overrides += [
            "actor_rollout_ref.model.lora_rank=" + str(recipe.peft.r),
            "actor_rollout_ref.model.lora_alpha=" + str(recipe.peft.alpha),
            "actor_rollout_ref.model.lora_dropout=" + str(recipe.peft.dropout),
        ]

    # Parallelism
    if recipe.parallel.tp > 1:
        overrides.append(f"actor_rollout_ref.rollout.tensor_model_parallel_size={recipe.parallel.tp}")
    if recipe.parallel.zero_stage:
        overrides.append(f"actor_rollout_ref.actor.fsdp_config.fsdp_size={recipe.parallel.zero_stage}")

    # Precision
    overrides.append(f"actor_rollout_ref.model.torch_dtype={recipe.precision}")

    # Chat template
    if recipe.target_template:
        overrides.append(f"actor_rollout_ref.model.chat_template={recipe.target_template}")

    # Reward function — pass the dotted reference; verl loads it in-process.
    if recipe.rl.reward_fn_ref:
        overrides.append(f"reward.custom_reward_function.path={recipe.rl.reward_fn_ref}")

    # Resume
    if recipe.io.resume_from:
        overrides.append(f"trainer.resume_from={recipe.io.resume_from}")

    # Preflight: cap to a single training step.
    if recipe.meta.get("__preflight__"):
        overrides.append(f"trainer.total_training_steps={int(recipe.meta.get('max_steps', 1))}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for line in overrides:
            f.write(line + "\n")
    return out_path


def load_overrides(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip() and not ln.startswith("#")]
