"""swift log parser — extracts loss / lr / grad_norm / kl from HF-Trainer-style lines.

swift wraps HuggingFace Trainer, so each step prints a JSON-ish dict
like::

    {'loss': 1.234, 'learning_rate': 5e-5, 'epoch': 0.01, 'step': 10}

There's also a `\\r`-progress-bar variant. We accept both.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Optional

from ...core.ir.metric import Metric
from ...core.logparser import LogParser, extract_kv, now_ts


_DICT_RE = re.compile(r"\{.*\}")


class SwiftLogParser(LogParser):
    framework = "swift"

    def parse_line(self, line: str) -> Optional[Metric]:
        line = line.strip()
        if not line:
            return None

        # Try strict JSON first (cheap), then python-literal-style dicts
        # (HF Trainer prints these), then fall back to key=value extraction.
        data = None
        m = _DICT_RE.search(line)
        if m is not None:
            blob = m.group(0)
            try:
                data = json.loads(blob)
            except json.JSONDecodeError:
                try:
                    data = ast.literal_eval(blob)
                except (ValueError, SyntaxError):
                    data = None

        if not isinstance(data, dict):
            kv = extract_kv(line)
            if "loss" not in kv and "reward" not in kv and "kl" not in kv:
                return None
            data = kv

        step = data.get("step") or data.get("global_step") or data.get("iter")
        if step is None:
            return None
        try:
            step = int(step)
        except (TypeError, ValueError):
            return None

        loss = _as_float(data.get("loss"))
        lr = _as_float(data.get("learning_rate") or data.get("lr"))
        gn = _as_float(data.get("grad_norm") or data.get("grad_norm_clipped"))
        epoch = _as_float(data.get("epoch"))
        kl = _as_float(data.get("kl") or data.get("kl_div"))
        reward = _as_float(data.get("reward") or data.get("reward_mean"))

        extra = {
            k: v
            for k, v in data.items()
            if k not in {"step", "global_step", "iter", "loss", "learning_rate",
                        "lr", "grad_norm", "grad_norm_clipped", "epoch", "kl",
                        "kl_div", "reward", "reward_mean"}
        }

        return Metric(
            step=step,
            timestamp=now_ts(),
            stage=self.stage,
            loss=loss,
            learning_rate=lr,
            grad_norm=gn,
            epoch=epoch,
            kl=kl,
            reward_mean=reward,
            extra=extra,
        )


def _as_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
