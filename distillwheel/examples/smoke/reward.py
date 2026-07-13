"""Rule reward used by the one-step verl GRPO smoke test."""

from __future__ import annotations

import re


def compute_score(data_source, solution_str, ground_truth, extra_info):
    """Return 1 when the response's final ``#### number`` matches the answer."""
    del data_source, extra_info
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", solution_str)
    return 1.0 if match and match.group(1) == str(ground_truth) else 0.0

