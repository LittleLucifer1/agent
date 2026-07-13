"""Recipe IR — declarative description of *what* to train.

A Recipe is the user-facing schema. Each backend translates it into its
native config (swift YAML, verl hydra overrides, ...).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Literal, Optional

from ..errors import IRValidationError

#: Bump whenever the Recipe schema changes in a breaking way.
RECIPE_SCHEMA_VERSION = 1

Stage = Literal["sft", "dpo", "kto", "grpo", "ppo", "rloo"]
Precision = Literal["bf16", "fp16", "fp8"]
PEFTType = Literal["lora", "qlora", "full"]

_STAGES = {"sft", "dpo", "kto", "grpo", "ppo", "rloo"}
_RL_STAGES = {"grpo", "ppo", "rloo"}
_PRECISIONS = {"bf16", "fp16", "fp8"}
_PEFT_TYPES = {"lora", "qlora", "full"}
_ROLLOUT_ENGINES = {"vllm", "sglang"}


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
    grad_accum: int = 32
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
        if self.stage not in _STAGES:
            raise IRValidationError(f"unknown recipe.stage={self.stage!r}; expected one of {sorted(_STAGES)}")
        if not isinstance(self.base_model, str) or not self.base_model.strip():
            raise IRValidationError("recipe.base_model is required")
        if self.precision not in _PRECISIONS:
            raise IRValidationError(
                f"unknown recipe.precision={self.precision!r}; expected one of {sorted(_PRECISIONS)}"
            )
        if not isinstance(self.io.output_dir, str) or not self.io.output_dir.strip():
            raise IRValidationError("io.output_dir must be a non-empty string")

        _require_number(self.train.epochs, "train.epochs", minimum=0.0, exclusive=True)
        for value, key in (
            (self.train.global_batch, "train.global_batch"),
            (self.train.micro_batch, "train.micro_batch"),
            (self.train.grad_accum, "train.grad_accum"),
            (self.train.max_len, "train.max_len"),
            (self.parallel.dp, "parallel.dp"),
            (self.parallel.tp, "parallel.tp"),
            (self.parallel.pp, "parallel.pp"),
            (self.io.save_steps, "io.save_steps"),
            (self.io.logging_steps, "io.logging_steps"),
        ):
            _require_positive_int(value, key)
        if not isinstance(self.train.seed, int) or isinstance(self.train.seed, bool):
            raise IRValidationError("train.seed must be an integer")
        if not isinstance(self.parallel.zero_stage, int) or isinstance(self.parallel.zero_stage, bool):
            raise IRValidationError("parallel.zero_stage must be an integer")
        if self.parallel.zero_stage not in (0, 1, 2, 3):
            raise IRValidationError("parallel.zero_stage must be one of 0, 1, 2, 3")

        effective_global = (
            self.train.micro_batch * self.train.grad_accum * self.parallel.dp
        )
        if self.train.global_batch != effective_global:
            raise IRValidationError(
                "train.global_batch must equal train.micro_batch * "
                f"train.grad_accum * parallel.dp ({effective_global}), "
                f"got {self.train.global_batch}"
            )

        _require_number(self.optim.lr, "optim.lr", minimum=0.0, exclusive=True)
        _require_number(self.optim.warmup_ratio, "optim.warmup_ratio", minimum=0.0, maximum=1.0)
        _require_number(self.optim.weight_decay, "optim.weight_decay", minimum=0.0)

        if self.peft is not None:
            if self.peft.type not in _PEFT_TYPES:
                raise IRValidationError(
                    f"unknown peft.type={self.peft.type!r}; expected one of {sorted(_PEFT_TYPES)}"
                )
            _require_positive_int(self.peft.r, "peft.r")
            _require_positive_int(self.peft.alpha, "peft.alpha")
            _require_number(self.peft.dropout, "peft.dropout", minimum=0.0)
            if float(self.peft.dropout) >= 1.0:
                raise IRValidationError("peft.dropout must be >= 0 and < 1")
            if not isinstance(self.peft.target_modules, list) or not all(
                isinstance(item, str) and item for item in self.peft.target_modules
            ):
                raise IRValidationError("peft.target_modules must be a list of non-empty strings")

        if self.stage in _RL_STAGES and self.rl is None:
            raise IRValidationError(f"recipe.rl is required for stage={self.stage}")
        if self.rl is not None:
            _require_number(self.rl.kl_coef, "rl.kl_coef", minimum=0.0)
            _require_number(self.rl.clip, "rl.clip", minimum=0.0, maximum=1.0, exclusive=True)
            _require_positive_int(self.rl.rollout_n, "rl.rollout_n")
            if self.rl.rollout_engine not in _ROLLOUT_ENGINES:
                raise IRValidationError(
                    f"unknown rl.rollout_engine={self.rl.rollout_engine!r}; "
                    f"expected one of {sorted(_ROLLOUT_ENGINES)}"
                )
            if self.rl.reward_fn_ref is not None and (
                not isinstance(self.rl.reward_fn_ref, str) or not self.rl.reward_fn_ref.strip()
            ):
                raise IRValidationError("rl.reward_fn_ref must be a non-empty string when set")

        if self.target_template is not None and not isinstance(self.target_template, str):
            raise IRValidationError("target_template must be a string when set")
        if self.backend_hint is not None and (
            not isinstance(self.backend_hint, str) or not self.backend_hint.strip()
        ):
            raise IRValidationError("backend_hint must be a non-empty string when set")
        if not isinstance(self.meta, dict):
            raise IRValidationError("recipe.meta must be a mapping")
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
        if not isinstance(d, Mapping):
            raise IRValidationError("recipe must be a mapping")
        data = dict(d)
        version = data.pop("_recipe_schema_version", RECIPE_SCHEMA_VERSION)
        if version != RECIPE_SCHEMA_VERSION:
            raise IRValidationError(
                f"unsupported recipe schema version: got={version}, "
                f"expected={RECIPE_SCHEMA_VERSION}. "
                "Upgrade the recipe or pin an older distillwheel."
            )

        allowed = {
            "stage", "base_model", "train", "optim", "io", "peft",
            "precision", "parallel", "rl", "target_template", "backend_hint", "meta",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise IRValidationError(f"unknown recipe field(s): {unknown}")

        def _mapping(key: str, *, required: bool = False) -> dict:
            value = data.get(key)
            if value is None:
                if required:
                    raise IRValidationError(f"recipe.{key} is required")
                return {}
            if not isinstance(value, Mapping):
                raise IRValidationError(f"recipe.{key} must be a mapping")
            return dict(value)

        def _optional(config_cls, key: str):
            value = data.get(key)
            if value is None:
                return None
            if not isinstance(value, Mapping):
                raise IRValidationError(f"recipe.{key} must be a mapping")
            return config_cls(**dict(value))

        try:
            recipe = cls(
                stage=data["stage"],
                base_model=data["base_model"],
                train=TrainConfig(**_mapping("train")),
                optim=OptimConfig(**_mapping("optim")),
                io=IOConfig(**_mapping("io", required=True)),
                peft=_optional(PEFTConfig, "peft"),
                precision=data.get("precision", "bf16"),
                parallel=ParallelConfig(**_mapping("parallel")),
                rl=_optional(RLConfig, "rl"),
                target_template=data.get("target_template"),
                backend_hint=data.get("backend_hint"),
                meta={} if data.get("meta") is None else data.get("meta"),
            )
        except KeyError as exc:
            raise IRValidationError(f"missing recipe field: {exc.args[0]}") from exc
        except TypeError as exc:
            raise IRValidationError(f"invalid recipe field: {exc}") from exc
        recipe.validate()
        return recipe

    def to_yaml(self, path: Any) -> None:
        import yaml

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, path: Any) -> "Recipe":
        import yaml

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except OSError as exc:
            raise IRValidationError(f"cannot read recipe yaml {path}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise IRValidationError(f"invalid recipe yaml {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise IRValidationError(f"recipe yaml must be a mapping: {path}")
        return cls.from_dict(raw)


def _require_positive_int(value: Any, key: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise IRValidationError(f"{key} must be a positive integer")


def _require_number(
    value: Any,
    key: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    exclusive: bool = False,
) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise IRValidationError(f"{key} must be a number")
    numeric = float(value)
    if minimum is not None:
        invalid = numeric <= minimum if exclusive else numeric < minimum
        if invalid:
            op = ">" if exclusive else ">="
            raise IRValidationError(f"{key} must be {op} {minimum}")
    if maximum is not None and numeric > maximum:
        raise IRValidationError(f"{key} must be <= {maximum}")
