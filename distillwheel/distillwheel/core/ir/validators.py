"""Per-task Sample validation and Recipe-stage compatibility checks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..errors import IRValidationError

if TYPE_CHECKING:  # pragma: no cover
    from .sample import MessageOrText, Sample

_STAGE_TASK_TYPE = {
    "sft": "sft",
    "dpo": "preference",
    "kto": "kto",
    "grpo": "rl_prompt",
    "ppo": "rl_prompt",
    "rloo": "rl_prompt",
    "opd": "rl_prompt",
}
_ROLES = {"system", "user", "assistant", "tool"}


def _require(cond: bool, msg: str, sample_id: str) -> None:
    if not cond:
        raise IRValidationError(f"sample {sample_id!r}: {msg}")


def _validate_message_list(value: Any, field_name: str, sample_id: str) -> None:
    # Import lazily to keep ``sample`` -> ``validators`` free of a module-level
    # cycle while still enforcing the public IR type (rather than accepting any
    # object that merely happens to have ``role`` and ``content`` attributes).
    from .sample import Message

    _require(isinstance(value, list) and bool(value),
             f"`{field_name}` must be a non-empty message list", sample_id)
    for index, message in enumerate(value):
        _require(isinstance(message, Message),
                 f"`{field_name}[{index}]` must be a Message", sample_id)
        _require(message.role in _ROLES,
                 f"`{field_name}[{index}].role` is invalid: {message.role!r}", sample_id)
        _require(isinstance(message.content, str),
                 f"`{field_name}[{index}].content` must be a string", sample_id)
        tool_calls = getattr(message, "tool_calls", None)
        _require(
            tool_calls is None or (
                isinstance(tool_calls, list)
                and all(isinstance(call, dict) for call in tool_calls)
            ),
            f"`{field_name}[{index}].tool_calls` must be a list of mappings",
            sample_id,
        )
        tool_call_id = getattr(message, "tool_call_id", None)
        _require(
            tool_call_id is None or isinstance(tool_call_id, str),
            f"`{field_name}[{index}].tool_call_id` must be a string",
            sample_id,
        )
        if tool_calls is not None:
            _require(
                message.role == "assistant",
                f"`{field_name}[{index}].tool_calls` is only valid for assistant messages",
                sample_id,
            )
        if tool_call_id is not None:
            _require(
                message.role == "tool" and bool(tool_call_id.strip()),
                f"`{field_name}[{index}].tool_call_id` requires a tool message and a non-empty id",
                sample_id,
            )
        if message.role == "tool":
            _require(
                isinstance(tool_call_id, str) and bool(tool_call_id.strip()),
                f"`{field_name}[{index}]` tool messages require `tool_call_id`",
                sample_id,
            )


def _has_content(value: "MessageOrText") -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list) and value:
        return any(
            hasattr(message, "content")
            and isinstance(message.content, str)
            and bool(message.content.strip())
            for message in value
        )
    return False


def _require_message_or_text(
    value: "MessageOrText",
    field_name: str,
    sample_id: str,
    task_name: str,
) -> None:
    if isinstance(value, str):
        _require(bool(value.strip()),
                 f"{task_name} requires non-empty `{field_name}`", sample_id)
        return
    if isinstance(value, list):
        _validate_message_list(value, field_name, sample_id)
        _require(_has_content(value),
                 f"{task_name} requires non-empty `{field_name}` content", sample_id)
        return
    _require(False, f"{task_name} requires `{field_name}` as text or messages", sample_id)


def validate_sample(s: "Sample") -> None:
    _require(isinstance(s.id, str) and bool(s.id.strip()), "id must be non-empty", s.id)
    _require(isinstance(s.meta, dict), "meta must be a mapping", s.id)
    _require(
        s.tools is None or (
            isinstance(s.tools, list)
            and all(isinstance(tool, dict) for tool in s.tools)
        ),
        "tools must be a list of mappings",
        s.id,
    )
    _require(
        s.images is None or (
            isinstance(s.images, list)
            and all(isinstance(image, str) and bool(image.strip()) for image in s.images)
        ),
        "images must be a list of non-empty strings",
        s.id,
    )

    if s.task_type == "sft":
        _validate_message_list(s.messages, "messages", s.id)
        assert s.messages is not None
        last = s.messages[-1]
        _require(last.role == "assistant",
                 "sft requires the last message to be from the assistant", s.id)
        _require(
            bool(last.content.strip()) or bool(last.tool_calls),
            "sft requires a non-empty final assistant response or tool call",
            s.id,
        )
    elif s.task_type == "preference":
        _require_message_or_text(s.prompt, "prompt", s.id, "preference")
        _require_message_or_text(s.chosen, "chosen", s.id, "preference")
        _require_message_or_text(s.rejected, "rejected", s.id, "preference")
    elif s.task_type == "kto":
        _require_message_or_text(s.prompt, "prompt", s.id, "kto")
        _require(isinstance(s.completion, str) and bool(s.completion.strip()),
                 "kto requires non-empty `completion`", s.id)
        valid_label = isinstance(s.label, bool) or (
            isinstance(s.label, (int, float))
            and not isinstance(s.label, bool)
            and s.label in (0, 1)
        )
        _require(valid_label, "kto requires `label` to be bool, 0, or 1", s.id)
    elif s.task_type == "rl_prompt":
        _require_message_or_text(s.prompt, "prompt", s.id, "rl_prompt")
    else:
        raise IRValidationError(f"sample {s.id!r}: unknown task_type={s.task_type!r}")


def validate_sample_for_stage(s: "Sample", stage: str) -> None:
    """Validate a sample and ensure its task type matches the Recipe stage."""
    validate_sample(s)
    expected = _STAGE_TASK_TYPE.get(stage)
    if expected is None:
        raise IRValidationError(f"unknown recipe stage while validating data: {stage!r}")
    _require(
        s.task_type == expected,
        f"task_type={s.task_type!r} is incompatible with stage={stage!r}; expected {expected!r}",
        s.id,
    )


__all__ = ["validate_sample", "validate_sample_for_stage"]
