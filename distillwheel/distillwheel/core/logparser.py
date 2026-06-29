"""LogParser base class + a couple of generic regex helpers.

Each backend subclasses ``LogParser`` and implements ``parse_line`` to
turn its stdout into normalized :class:`Metric` instances.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from typing import Iterable, Iterator, Optional

from .ir.metric import Metric


class LogParser(ABC):
    """Extract :class:`Metric` from a stream of stdout lines."""

    stage: str = "unknown"

    @abstractmethod
    def parse_line(self, line: str) -> Optional[Metric]:
        """Return a Metric for recognised lines, else ``None``."""

    def parse_stream(self, lines: Iterable[str]) -> Iterator[Metric]:
        for line in lines:
            m = self.parse_line(line)
            if m is not None:
                yield m


# ---------- helpers ----------

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def extract_kv(line: str) -> dict:
    """Heuristic ``key=value`` / ``key: value`` extractor.

    Used by the swift parser as a cheap default. Returns a dict whose
    values are floats where parsable, strings otherwise.
    """
    out = {}
    # 'key=value' and 'key: value' both common in HF/Trainer logs
    for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_./]*)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|\".*?\"|'.*?'|\S+)", line):
        k, v = m.group(1), m.group(2)
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v.strip("\"'")
    return out


def now_ts() -> float:
    return time.time()


def first_float(s: str) -> Optional[float]:
    m = _NUMBER_RE.search(s)
    if m is None:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None
