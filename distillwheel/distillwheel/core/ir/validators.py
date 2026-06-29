"""Per-task-type Sample validation.

Kept in its own module so ``Sample.validate`` can import lazily and
avoid an import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..errors import IRValidationError

if TYPE_CHECKING:  # pragma: no cover
    from .sample import Sample


def _require(cond: bool, msg: str, sample_id: str) -> None:
    if not cond:
        raise IRValidationError(f"sample {sample_id!r}: {msg}")


def validate_sample(s: "Sample") -> None:
    _require(bool(s.id), "id must be non-empty", s.id)
    if s.task_type == "sft":
        _require(s.messages is not None and len(s.messages) > 0,
                 "sft requires non-empty `messages`", s.id)
        # last assistant message must exist for supervised training
        roles = [m.role for m in (s.messages or [])]
        _require("assistant" in roles, "sft requires at least one assistant message", s.id)
    elif s.task_type == "preference":
        _require(s.prompt is not None, "preference requires `prompt`", s.id)
        _require(s.chosen is not None, "preference requires `chosen`", s.id)
        _require(s.rejected is not None, "preference requires `rejected`", s.id)
    elif s.task_type == "kto":
        _require(s.prompt is not None, "kto requires `prompt`", s.id)
        _require(s.completion is not None, "kto requires `completion`", s.id)
        _require(isinstance(s.label, (bool, float, int)),
                 "kto requires boolean/float `label`", s.id)
    elif s.task_type == "rl_prompt":
        _require(s.prompt is not None, "rl_prompt requires `prompt`", s.id)
    else:
        raise IRValidationError(f"sample {s.id!r}: unknown task_type={s.task_type!r}")
