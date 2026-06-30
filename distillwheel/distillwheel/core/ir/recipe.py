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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal, Optional

from ..errors import IRValidationError

#: Bump whenever the Recipe schema changes in a breaking way.
RECIPE_SCHEMA_VERSION = 1

Stage = Literal["sft", "dpo", "kto", "grpo", "ppo", "rloo", "opd"]
Precision = Literal["bf16", "fp16", "fp8"]
PEFTType = Literal["lora", "qlora", "full"]


@dataclass
class PEFTConfig:
    """参数高效微调配置。type="full" 表示全参训练(不使用 PEFT)。"""
    type: PEFTType = "lora"                      # lora / qlora / full
    r: int = 16                                  # LoRA 秩
    alpha: int = 32                              # LoRA alpha (缩放 = alpha/r)
    target_modules: List[str] = field(default_factory=list)  # 应用 LoRA 的模块名
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
    """训练超参。有效 batch size = micro_batch × grad_accum × dp。"""
    epochs: float = 1.0                          # 训练轮数
    global_batch: int = 32                       # 全局 batch size
    micro_batch: int = 1                         # 单卡单步 batch size
    grad_accum: int = 1                          # 梯度累积步数
    max_len: int = 4096                          # 最大序列长度
    seed: int = 42                               # 随机种子


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
