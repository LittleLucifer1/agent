"""Normalized training-metric schema.

Every backend's LogParser emits :class:`Metric` instances, which the
pipeline serializes to ``metrics.jsonl``. Downstream dashboards read
only that file, so they don't need to know which framework produced it.

字段映射
========

各 backend 的原始日志键名不同，LogParser 负责映射到统一字段:

+-----------------------+---------------------+-------------------------------+
| 统一字段              | swift 原始键         | verl 原始键                    |
+=======================+=====================+===============================+
| loss                  | loss                | actor/loss 或 loss             |
| eval_loss             | eval_loss           | eval/loss                     |
| learning_rate         | learning_rate / lr  | actor/lr / lr                 |
| grad_norm             | grad_norm           | actor/grad_norm               |
| kl                    | kl / kl_div         | actor/kl / approx_kl          |
| reward_mean           | reward / reward_mean| reward/mean                   |
| reward_std            | -                   | reward/std                    |
| policy_loss           | -                   | actor/pg_loss                 |
| value_loss            | -                   | critic/loss                   |
| entropy               | -                   | actor/entropy                 |
| clip_fraction         | -                   | actor/clip_fraction           |
| response_length_mean  | -                   | response/length_mean          |
+-----------------------+---------------------+-------------------------------+

metrics.jsonl 示例
==================

**SFT** — 每行一个 JSON, 只有通用字段有值::

    {"step":10,  "timestamp":1719600000.0, "stage":"sft", "loss":2.31, "learning_rate":5e-5, "grad_norm":1.2, "epoch":0.01, ...}
    {"step":20,  "timestamp":1719600060.0, "stage":"sft", "loss":1.87, "learning_rate":4.8e-5, "grad_norm":0.9, "epoch":0.02, ...}

**GRPO** — RL 字段也会被填充::

    {"step":10, "timestamp":1719600000.0, "stage":"grpo", "loss":0.21, "kl":0.013,
     "reward_mean":0.81, "reward_std":0.12, "policy_loss":0.18, "entropy":2.3,
     "clip_fraction":0.15, "response_length_mean":256.3, ...}
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Optional


@dataclass
class Metric:
    """一个训练步的归一化指标快照。"""

    # ── 上下文 (每条必填) ────────────────────────────────────────────
    step: int                                    # 全局训练步数
    timestamp: float                             # Unix 时间戳 (由 LogParser 在解析时填入)
    stage: str                                   # 训练阶段: sft / dpo / grpo / ...

    # ── 通用训练指标 (SFT / DPO / KTO / RL 都可能产出) ──────────────
    loss: Optional[float] = None                 # 训练 loss
    eval_loss: Optional[float] = None            # 验证集 loss (如果有 eval 阶段)
    learning_rate: Optional[float] = None        # 当前学习率
    grad_norm: Optional[float] = None            # 梯度范数 (监控训练稳定性)
    epoch: Optional[float] = None                # 当前 epoch 进度 (小数, 如 0.5 = 半个 epoch)

    # ── RL 专用指标 (GRPO / PPO / RLOO / OPD) ──────────────────────
    reward_mean: Optional[float] = None          # 平均奖励 — 核心优化目标
    reward_std: Optional[float] = None           # 奖励标准差 — 衡量奖励分布
    kl: Optional[float] = None                   # KL 散度 — 策略偏离参考模型的程度
    policy_loss: Optional[float] = None          # 策略 loss (actor loss)
    value_loss: Optional[float] = None           # 价值函数 loss (critic loss, PPO 专用)
    entropy: Optional[float] = None              # 策略熵 — 过低意味着模型 collapse
    clip_fraction: Optional[float] = None        # PPO/GRPO clip 比例 — 过高说明训练不稳定
    response_length_mean: Optional[float] = None # 平均回答长度 — 突增/突降可能是 reward hacking

    # ── 扩展 ────────────────────────────────────────────────────────
    extra: dict = field(default_factory=dict)    # 框架特有的指标透传, 不在上面列表中的都放这里

    def to_dict(self) -> dict:
        # Metrics are observability data. A framework-specific value such as
        # ``nan``, bytes, a set, or a Path must not crash an otherwise healthy
        # training run or produce non-standard JSON. Keep the normalized file
        # strictly JSON-compatible, using ``null`` for non-finite floats.
        return {
            item.name: _json_safe(getattr(self, item.name), seen=set())
            for item in fields(self)
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)


def _json_safe(value, *, seen: set[int]):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, os.PathLike):
        return os.fspath(value)

    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        identity = id(value)
        if identity in seen:
            return "<recursive>"
        seen.add(identity)
        try:
            if isinstance(value, Mapping):
                return {
                    str(key): _json_safe(item, seen=seen)
                    for key, item in value.items()
                }
            items = value
            if isinstance(value, (set, frozenset)):
                items = sorted(value, key=repr)
            return [_json_safe(item, seen=seen) for item in items]
        finally:
            seen.remove(identity)

    try:
        return str(value)
    except Exception:  # pragma: no cover - pathological third-party object
        return f"<unserializable {type(value).__name__}>"
