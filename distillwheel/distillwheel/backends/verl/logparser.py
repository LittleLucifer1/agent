"""verl log parser.

verl prints structured per-step lines that look roughly like::

    step=10 actor/loss=0.21 actor/kl=0.013 critic/loss=0.045 \
        reward/mean=0.81 reward/std=0.12 actor/pg_loss=0.18 actor/entropy=2.3

We extract those with a generous regex and map slash-prefixed keys to
the unified :class:`Metric` schema. See ``docs/metric_mapping.md``.
"""

from __future__ import annotations

import re
from typing import Optional

from ...core.ir.metric import Metric
from ...core.logparser import LogParser, now_ts

_KV = re.compile(r"([A-Za-z_][A-Za-z0-9_./]*)=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
_STEP = re.compile(r"\b(?:step|global_step|iteration)=(\d+)\b")


class VerlLogParser(LogParser):
    framework = "verl"

    def parse_line(self, line: str) -> Optional[Metric]:
        line = line.strip()
        if not line:
            return None

        step_match = _STEP.search(line)
        if step_match is None:
            return None
        step = int(step_match.group(1))

        kvs = {k: float(v) for k, v in _KV.findall(line)}
        if not kvs:
            return None

        return Metric(
            step=step,
            timestamp=now_ts(),
            stage=self.stage,
            loss=_first(kvs, "actor/loss", "loss"),
            learning_rate=_first(kvs, "actor/lr", "learning_rate", "lr"),
            grad_norm=_first(kvs, "actor/grad_norm", "grad_norm"),
            epoch=_first(kvs, "epoch"),
            kl=_first(kvs, "actor/kl", "approx_kl", "kl"),
            reward_mean=_first(kvs, "reward/mean", "reward_mean"),
            reward_std=_first(kvs, "reward/std", "reward_std"),
            policy_loss=_first(kvs, "actor/pg_loss", "actor/policy_loss"),
            value_loss=_first(kvs, "critic/loss", "value_loss"),
            entropy=_first(kvs, "actor/entropy", "entropy"),
            extra={k: v for k, v in kvs.items() if k not in _CONSUMED},
        )


_CONSUMED = {
    "step", "global_step", "iteration", "epoch",
    "actor/loss", "loss",
    "actor/lr", "learning_rate", "lr",
    "actor/grad_norm", "grad_norm",
    "actor/kl", "approx_kl", "kl",
    "reward/mean", "reward_mean",
    "reward/std", "reward_std",
    "actor/pg_loss", "actor/policy_loss",
    "critic/loss", "value_loss",
    "actor/entropy", "entropy",
}


def _first(d: dict, *keys: str) -> Optional[float]:
    for k in keys:
        if k in d:
            return d[k]
    return None
