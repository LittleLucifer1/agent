"""Parse ms-swift 4.4 / Hugging Face Trainer metric log lines."""

from __future__ import annotations

import ast
import json
import re
from typing import Optional

from ...core.ir.metric import Metric
from ...core.logparser import LogParser, extract_kv, now_ts


_DICT_RE = re.compile(r"\{.*\}")
_METRIC_KEYS = {
    "loss",
    "train/loss",
    "train_loss",
    "reward",
    "reward_mean",
    "rewards/mean",
    "reward/mean",
    "kl",
    "kl_div",
    "objective/kl",
}
_STEP_KEYS = ("step", "global_step", "iter", "global_step/max_steps")
_LOSS_KEYS = ("loss", "train/loss", "train_loss")
_LR_KEYS = ("learning_rate", "train/learning_rate", "lr")
_GRAD_KEYS = ("grad_norm", "train/grad_norm", "grad_norm_clipped")
_EPOCH_KEYS = ("epoch", "train/epoch")
_KL_KEYS = ("kl", "kl_div", "objective/kl", "train/kl")
_REWARD_MEAN_KEYS = ("reward", "reward_mean", "rewards/mean", "reward/mean")
_REWARD_STD_KEYS = ("reward_std", "rewards/std", "reward/std")
_POLICY_LOSS_KEYS = ("policy_loss", "loss/policy", "actor/policy_loss")
_VALUE_LOSS_KEYS = ("value_loss", "loss/value", "critic/loss")
_ENTROPY_KEYS = ("entropy", "policy/entropy", "actor/entropy")
_CONSUMED_KEYS = set().union(
    _STEP_KEYS,
    _LOSS_KEYS,
    _LR_KEYS,
    _GRAD_KEYS,
    _EPOCH_KEYS,
    _KL_KEYS,
    _REWARD_MEAN_KEYS,
    _REWARD_STD_KEYS,
    _POLICY_LOSS_KEYS,
    _VALUE_LOSS_KEYS,
    _ENTROPY_KEYS,
)


class SwiftLogParser(LogParser):
    framework = "swift"

    def parse_line(self, line: str) -> Optional[Metric]:
        line = line.strip()
        if not line:
            return None

        # ms-swift prints Python-literal dictionaries, often after an INFO
        # prefix.  JSON and key=value forms are accepted as well.
        data = None
        match = _DICT_RE.search(line)
        if match is not None:
            blob = match.group(0)
            try:
                data = json.loads(blob)
            except json.JSONDecodeError:
                try:
                    data = ast.literal_eval(blob)
                except (ValueError, SyntaxError):
                    data = None

        if not isinstance(data, dict):
            kv = extract_kv(line)
            if not any(key in kv for key in _METRIC_KEYS):
                return None
            data = kv

        step = _extract_step(data)
        if step is None:
            return None

        return Metric(
            step=step,
            timestamp=now_ts(),
            stage=self.stage,
            loss=_first_float(data, *_LOSS_KEYS),
            learning_rate=_first_float(data, *_LR_KEYS),
            grad_norm=_first_float(data, *_GRAD_KEYS),
            epoch=_first_float(data, *_EPOCH_KEYS),
            kl=_first_float(data, *_KL_KEYS),
            reward_mean=_first_float(data, *_REWARD_MEAN_KEYS),
            reward_std=_first_float(data, *_REWARD_STD_KEYS),
            policy_loss=_first_float(data, *_POLICY_LOSS_KEYS),
            value_loss=_first_float(data, *_VALUE_LOSS_KEYS),
            entropy=_first_float(data, *_ENTROPY_KEYS),
            extra={key: value for key, value in data.items() if key not in _CONSUMED_KEYS},
        )


def _extract_step(data: dict) -> Optional[int]:
    for key in ("step", "global_step", "iter"):
        if key not in data:
            continue
        try:
            return int(data[key])
        except (TypeError, ValueError):
            pass

    # Current ms-swift progress dictionaries use e.g. ``'1/840'``.
    value = data.get("global_step/max_steps")
    if isinstance(value, str):
        match = re.match(r"^\s*(\d+)\s*/\s*\d+\s*$", value)
        if match:
            return int(match.group(1))
    elif value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    return None


def _first_float(data: dict, *keys: str) -> Optional[float]:
    for key in keys:
        if key in data:
            return _as_float(data[key])
    return None


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
