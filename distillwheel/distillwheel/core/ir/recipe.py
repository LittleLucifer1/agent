"""Recipe IR — declarative description of *what* to train.

A Recipe is the user-facing schema. Each backend translates it into its
native config (swift YAML, verl hydra overrides, ...)
=============

不同 stage 会被路由到不同 backend (见 ``core/router.py``)::

    SFT / DPO / KTO  →  swift backend
    GRPO / PPO / RLOO / OPD  →  verl backend

用户也可以通过 ``backend_hint`` 强制指定。

Sub-config 组成
================

一个 Recipe 由以下子配置组合而成::

    Recipe
    ├── train:     TrainConfig      训练超参 (epoch, batch, 序列长度 ...)
    ├── optim:     OptimConfig      优化器 (lr, scheduler, warmup ...)
    ├── io:        IOConfig         输入输出 (output_dir, save/log 频率 ...)
    ├── peft:      PEFTConfig       参数高效微调 (LoRA/QLoRA/全参)
    ├── parallel:  ParallelConfig   并行策略 (DP/TP/PP/ZeRO)
    └── rl:        RLConfig         强化学习专用 (KL, clip, rollout ...) — 仅 RL stage 需要

SFT (LoRA)
==========================
    stage: sft
    base_model: Qwen/Qwen2.5-7B
    target_template: qwen                       # 强制 chat template, 保证行为一致

    train:
      epochs: 3
      global_batch: 32                          # 有效 batch = micro_batch × grad_accum × dp
      micro_batch: 2
      grad_accum: 8
      max_len: 4096
      seed: 42

    optim:
      lr: 1e-4
      scheduler: cosine
      warmup_ratio: 0.05

    peft:
      type: lora
      r: 64
      alpha: 128
      target_modules: [q_proj, k_proj, v_proj, o_proj]
      dropout: 0.05

    io:
      output_dir: outputs/qwen7b-sft-lora
      save_steps: 200
      logging_steps: 10

DPO
===================
    stage: dpo
    base_model: Qwen/Qwen2.5-7B
    target_template: qwen

    train:
      epochs: 1
      global_batch: 16
      micro_batch: 1
      grad_accum: 16
      max_len: 2048

    optim:
      lr: 5e-6 
      scheduler: cosine
      warmup_ratio: 0.1

    peft:
      type: lora
      r: 32
      alpha: 64

    io:
      output_dir: outputs/qwen7b-dpo

GRPO (强化学习)
==============================
    stage: grpo
    base_model: Qwen/Qwen2.5-7B
    target_template: qwen

    train:
      epochs: 1
      global_batch: 64
      micro_batch: 1
      max_len: 4096

    optim:
      lr: 1e-6                                  # RL 学习率通常很小
      scheduler: cosine

    rl:                                          # ← RL stage 必填
      kl_coef: 0.05                             # KL 散度惩罚系数
      clip: 0.2                                 # PPO clip ratio
      rollout_n: 4                              # 每个 prompt 采样几条回答
      rollout_engine: vllm                      # 推理引擎: vllm 或 sglang
      reward_fn_ref: my_rewards:math_reward     # 奖励函数引用 "module:fn_name"

    parallel:
      tp: 2                                     # tensor parallel (需要多卡)

    peft:
      type: full                                # RL 一般全参训练

    io:
      output_dir: outputs/qwen7b-grpo
      save_steps: 100
"""

from __future__ import annotations

import dataclasses
import math
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
    """参数高效微调配置。type="full" 表示全参训练(不使用 PEFT)。"""
    type: PEFTType = "lora"                      # lora / qlora / full
    r: int = 16                                  # LoRA 秩
    alpha: int = 32                              # LoRA alpha (缩放 = alpha/r)
    target_modules: list[str] = field(default_factory=list)  # 应用 LoRA 的模块名
    dropout: float = 0.0                         # LoRA dropout


@dataclass
class OptimConfig:
    """优化器配置。"""
    lr: float = 5e-5                             # 学习率
    scheduler: str = "cosine"                    # lr scheduler 类型
    warmup_ratio: float = 0.03                   # warmup 占总步数的比例
    weight_decay: float = 0.0                    # 权重衰减


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
    """分布式并行策略。"""
    dp: int = 1                                  # 数据并行度
    tp: int = 1                                  # 张量并行度 (模型切分到多卡)
    pp: int = 1                                  # 流水线并行度
    zero_stage: int = 0                          # DeepSpeed ZeRO stage (0/1/2/3)


@dataclass
class RLConfig:
    """强化学习专用配置。仅当 stage 为 grpo/ppo/rloo/opd 时需要。"""
    kl_coef: float = 0.05                        # KL 散度惩罚系数 (约束策略不偏离参考模型太远)
    clip: float = 0.2                            # PPO/GRPO clip ratio
    rollout_n: int = 4                           # 每个 prompt 采样的回答数
    rollout_engine: Literal["vllm", "sglang"] = "vllm"  # rollout 推理引擎
    reward_fn_ref: Optional[str] = None          # 奖励函数引用, 格式 "module.path:fn_name"


@dataclass
class IOConfig:
    """输入输出配置。"""
    output_dir: str                              # 输出根目录
    save_steps: int = 500                        # 每 N 步保存 checkpoint
    logging_steps: int = 10                      # 每 N 步记录指标
    resume_from: Optional[str] = None            # 断点续训的 checkpoint 路径


@dataclass
class Recipe:
    """训练配方 — 声明 "训练什么、怎么训练"，由 backend adapter 翻译成框架原生配置。"""

    # ── 必填 ──────────────────────────────────────────────────
    stage: Stage                                 # 训练方法: sft/dpo/kto/grpo/ppo/rloo/opd
    base_model: str                              # HuggingFace 模型路径或本地路径
    train: TrainConfig                           # 训练超参
    optim: OptimConfig                           # 优化器配置
    io: IOConfig                                 # 输入输出配置

    # ── 可选子配置 ───────────────────────────────────────────────────
    peft: Optional[PEFTConfig] = None            # LoRA/QLoRA 配置, None = 全参训练
    precision: Precision = "bf16"                # 训练精度
    parallel: ParallelConfig = field(default_factory=ParallelConfig)  # 并行策略
    rl: Optional[RLConfig] = None                # RL 配置 — grpo/ppo/rloo/opd 必填

    # ── 路由与兼容 ───────────────────────────────────────────────────
    target_template: Optional[str] = None        # 强制 chat template 名称, 保证跨框架行为一致
    backend_hint: Optional[str] = None           # 强制指定 backend (跳过默认路由)
    meta: dict = field(default_factory=dict)     # 透传给 backend 的自定义字段

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
        if not isinstance(self.optim.scheduler, str) or not self.optim.scheduler.strip():
            raise IRValidationError("optim.scheduler must be a non-empty string")
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
        if isinstance(self.target_template, str) and not self.target_template.strip():
            raise IRValidationError("target_template must be a non-empty string when set")
        if self.backend_hint is not None and (
            not isinstance(self.backend_hint, str) or not self.backend_hint.strip()
        ):
            raise IRValidationError("backend_hint must be a non-empty string when set")
        if self.io.resume_from is not None and (
            not isinstance(self.io.resume_from, str) or not self.io.resume_from.strip()
        ):
            raise IRValidationError("io.resume_from must be a non-empty string when set")
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
        if type(version) is not int or version != RECIPE_SCHEMA_VERSION:
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
        except (OSError, UnicodeError) as exc:
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
    if not math.isfinite(numeric):
        raise IRValidationError(f"{key} must be finite")
    if minimum is not None:
        invalid = numeric <= minimum if exclusive else numeric < minimum
        if invalid:
            op = ">" if exclusive else ">="
            raise IRValidationError(f"{key} must be {op} {minimum}")
    if maximum is not None and numeric > maximum:
        raise IRValidationError(f"{key} must be <= {maximum}")
