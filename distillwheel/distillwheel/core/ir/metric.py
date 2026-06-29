"""Normalized training-metric schema.

Every backend's LogParser emits :class:`Metric` instances, which the
pipeline serializes to ``metrics.jsonl``. Downstream dashboards read
only that file, so they don't need to know which framework produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Metric:
    step: int
    timestamp: float
    stage: str
    # generic
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    grad_norm: Optional[float] = None
    epoch: Optional[float] = None
    # RL-specific (unified keys; per-backend mapping in docs/metric_mapping.md)
    reward_mean: Optional[float] = None
    reward_std: Optional[float] = None
    kl: Optional[float] = None
    policy_loss: Optional[float] = None
    value_loss: Optional[float] = None
    entropy: Optional[float] = None
    # escape hatch for framework-specific extras
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
