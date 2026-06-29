"""Recipe IR — declarative description of *what* to train.

A Recipe is the user-facing schema. Each backend translates it into its
native config (swift YAML, verl hydra overrides, ...).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from ..errors import IRValidationError

#: Bump whenever the Recipe schema changes in a breaking way.
RECIPE_SCHEMA_VERSION = 1

Stage = Literal["sft", "dpo", "kto", "grpo", "ppo", "rloo", "opd"]
Precision = Literal["bf16", "fp16", "fp8"]
PEFTType = Literal["lora", "qlora", "full"]


@dataclass
class PEFTConfig:
    type: PEFTType = "lora"
    r: int = 16
    alpha: int = 32
    target_modules: list = field(default_factory=list)
    dropout: float = 0.0


@dataclass
class OptimConfig:
    lr: float = 5e-5
    scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0


@dataclass
class TrainConfig:
    epochs: float = 1.0
    global_batch: int = 32
    micro_batch: int = 1
    grad_accum: int = 1
    max_len: int = 4096
    seed: int = 42


@dataclass
class ParallelConfig:
    dp: int = 1
    tp: int = 1
    pp: int = 1
    zero_stage: int = 0


@dataclass
class RLConfig:
    kl_coef: float = 0.05
    clip: float = 0.2
    rollout_n: int = 4
    rollout_engine: Literal["vllm", "sglang"] = "vllm"
    reward_fn_ref: Optional[str] = None  # "module.path:fn_name"


@dataclass
class IOConfig:
    output_dir: str
    save_steps: int = 500
    logging_steps: int = 10
    resume_from: Optional[str] = None


@dataclass
class Recipe:
    stage: Stage
    base_model: str
    train: TrainConfig
    optim: OptimConfig
    io: IOConfig
    peft: Optional[PEFTConfig] = None
    precision: Precision = "bf16"
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    rl: Optional[RLConfig] = None
    target_template: Optional[str] = None       # forced chat template name
    backend_hint: Optional[str] = None          # user override for routing
    meta: dict = field(default_factory=dict)

    # ---------- validation ----------

    def validate(self) -> None:
        if not self.base_model:
            raise IRValidationError("recipe.base_model is required")
        if self.stage in ("grpo", "ppo", "rloo", "opd"):
            if self.rl is None:
                raise IRValidationError(
                    f"recipe.rl is required for stage={self.stage}")
        if self.train.global_batch < self.train.micro_batch:
            raise IRValidationError("global_batch must be >= micro_batch")
        if self.train.epochs <= 0:
            raise IRValidationError("train.epochs must be > 0")
        if not self.target_template:
            # Strong consistency: keep chat template explicit. Adapters
            # still treat None as "framework default" but emit a warning.
            # We treat missing as a soft validation issue, not fatal.
            pass

    # ---------- serialization ----------

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["_recipe_schema_version"] = RECIPE_SCHEMA_VERSION
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Recipe":
        version = d.pop("_recipe_schema_version", RECIPE_SCHEMA_VERSION)
        if version != RECIPE_SCHEMA_VERSION:
            raise IRValidationError(
                f"unsupported recipe schema version: got={version}, "
                f"expected={RECIPE_SCHEMA_VERSION}. "
                "Upgrade the recipe or pin an older distillwheel."
            )

        def _opt(cls_, key):
            v = d.get(key)
            return cls_(**v) if v is not None else None

        return cls(
            stage=d["stage"],
            base_model=d["base_model"],
            train=TrainConfig(**(d.get("train") or {})),
            optim=OptimConfig(**(d.get("optim") or {})),
            io=IOConfig(**d["io"]),
            peft=_opt(PEFTConfig, "peft"),
            precision=d.get("precision", "bf16"),
            parallel=ParallelConfig(**(d.get("parallel") or {})),
            rl=_opt(RLConfig, "rl"),
            target_template=d.get("target_template"),
            backend_hint=d.get("backend_hint"),
            meta=d.get("meta", {}),
        )

    def to_yaml(self, path: Any) -> None:
        import yaml

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, path: Any) -> "Recipe":
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise IRValidationError(f"recipe yaml must be a mapping: {path}")
        recipe = cls.from_dict(raw)
        recipe.validate()
        return recipe
