"""Parse VERL 0.8 console metrics, including Ray-prefixed output."""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from ...core.ir.metric import Metric
from ...core.logparser import LogParser, now_ts

_NUMBER = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|inf|nan)"
_KV = re.compile(
    rf"([A-Za-z_][A-Za-z0-9_./()@-]*)\s*[:=]\s*({_NUMBER})",
    re.IGNORECASE,
)
_STEP = re.compile(r"\b(?:step|global_step|iteration)\s*[:=]\s*(\d+)\b")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_RAY_PREFIX = re.compile(r"^(?:\([^\n)]*\bpid=\d+\)\s*)+")


class VerlLogParser(LogParser):
    framework = "verl"

    def parse_line(self, line: str) -> Optional[Metric]:
        line = _RAY_PREFIX.sub("", _ANSI.sub("", line.strip()))
        if not line:
            return None
        step_match = _STEP.search(line)
        if step_match is None:
            return None

        values: Dict[str, float] = {}
        for key, raw in _KV.findall(line):
            try:
                values[key] = float(raw)
            except ValueError:  # Defensive: the regex already limits values.
                continue
        if not values:
            return None

        consumed = {"step", "global_step", "iteration"}

        def pick(*keys: str, prefixes: Tuple[str, ...] = ()) -> Optional[float]:
            for key in keys:
                if key in values:
                    consumed.add(key)
                    return values[key]
            for prefix in prefixes:
                for key, value in values.items():
                    if key.startswith(prefix):
                        consumed.add(key)
                        return value
            return None

        policy_loss = pick("actor/pg_loss", "actor/policy_loss")
        loss = pick("actor/loss", "loss")
        if loss is None:
            loss = policy_loss

        return Metric(
            step=int(step_match.group(1)),
            timestamp=now_ts(),
            stage=self.stage,
            loss=loss,
            learning_rate=pick(
                "actor/lr", "learning_rate", "lr", prefixes=("actor/lr(",)
            ),
            grad_norm=pick("actor/grad_norm", "grad_norm"),
            epoch=pick("training/epoch", "epoch"),
            kl=pick("actor/ppo_kl", "actor/kl", "approx_kl", "kl"),
            reward_mean=pick(
                "critic/rewards/mean", "critic/score/mean", "reward/mean", "reward_mean"
            ),
            reward_std=pick(
                "critic/rewards/std", "critic/score/std", "reward/std", "reward_std"
            ),
            policy_loss=policy_loss,
            value_loss=pick("critic/vf_loss", "critic/loss", "value_loss"),
            entropy=pick("actor/entropy_loss", "actor/entropy", "entropy"),
            extra={key: value for key, value in values.items() if key not in consumed},
        )
