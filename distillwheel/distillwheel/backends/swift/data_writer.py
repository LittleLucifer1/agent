"""Translate Sample IR rows into ms-swift 4.4 JSONL records.

The preference schema follows ms-swift's standard form: the chosen answer is
the final assistant message in ``messages`` and ``rejected_response`` carries
the rejected continuation.  DistillWheel's OpenAI-style tool calls are
translated into ms-swift's ``tool_call`` / ``tool_response`` message roles;
ms-swift's dataset preprocessor otherwise discards ``tool_calls`` and
``tool_call_id`` fields from message dictionaries.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable, List

from ...core.errors import IRValidationError
from ...core.ir.sample import Message, Sample
from ...core.ir.validators import validate_sample_for_stage


def _message_dict(value) -> dict:
    if isinstance(value, Message):
        return value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    raise IRValidationError(
        f"expected Message or mapping, got {type(value).__name__}"
    )


def _tool_call_content(call: Mapping) -> str:
    """Return one ms-swift tool-call payload from an OpenAI-style call."""
    if call.get("type", "function") != "function":
        raise IRValidationError("swift supports only function tool calls")

    function = call.get("function")
    if not isinstance(function, Mapping):
        raise IRValidationError("tool call must contain a function mapping")
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise IRValidationError("tool call function.name must be non-empty")

    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise IRValidationError(
                f"tool call arguments for {name!r} must be valid JSON"
            ) from exc
    if not isinstance(arguments, Mapping):
        raise IRValidationError(
            f"tool call arguments for {name!r} must encode a JSON object"
        )

    return json.dumps(
        {"name": name, "arguments": dict(arguments)},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _swift_message_dicts(value) -> List[dict]:
    """Convert one IR/OpenAI message into one or more ms-swift messages."""
    message = _message_dict(value)
    role = message.get("role")
    content = message.get("content", "")
    if not isinstance(content, str):
        raise IRValidationError("message content must be text for the swift backend")

    if role == "assistant" and message.get("tool_calls") is not None:
        tool_calls = message["tool_calls"]
        if not isinstance(tool_calls, list) or not tool_calls:
            raise IRValidationError("assistant tool_calls must be a non-empty list")
        result: List[dict] = []
        # Some providers allow explanatory text before a tool invocation.  It
        # remains a normal assistant turn in ms-swift.
        if content:
            result.append({"role": "assistant", "content": content})
        for call in tool_calls:
            if not isinstance(call, Mapping):
                raise IRValidationError("each tool call must be a mapping")
            result.append({"role": "tool_call", "content": _tool_call_content(call)})
        return result

    if role == "tool":
        return [{"role": "tool_response", "content": content}]
    if role not in {"system", "user", "assistant", "tool_call", "tool_response"}:
        raise IRValidationError(f"unsupported swift message role: {role!r}")

    result = {"role": role, "content": content}
    # These are the only optional message fields retained by ms-swift 4.4's
    # preprocessor.  They are accepted for mapping-based callers even though
    # the current Message IR does not expose them yet.
    for key in ("loss", "loss_scale"):
        if key in message:
            result[key] = message[key]
    return [result]


def _to_messages(value) -> List[dict]:
    if value is None:
        return []
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, (Message, Mapping)):
        return _swift_message_dicts(value)
    result: List[dict] = []
    for message in value:
        result.extend(_swift_message_dicts(message))
    return result


def _chosen_messages(value) -> List[dict]:
    if isinstance(value, str):
        return [{"role": "assistant", "content": value}]
    messages = _to_messages(value)
    if not messages or messages[-1].get("role") != "assistant":
        raise IRValidationError(
            "a structured DPO chosen response must be non-empty and end with "
            "an assistant message"
        )
    return messages


def _rejected_response(value):
    """Preserve text or a structured rejected continuation for ms-swift."""
    if isinstance(value, str):
        return value

    messages = _to_messages(value)
    if not messages or messages[-1].get("role") != "assistant":
        raise IRValidationError(
            "a structured DPO rejected response must be non-empty and end "
            "with an assistant message"
        )
    return messages


def _sample_extras(sample: Sample) -> dict:
    extras = {}
    if sample.tools is not None:
        extras["tools"] = sample.tools
    if sample.images is not None:
        extras["images"] = sample.images
    return extras


def _kto_label(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise IRValidationError("KTO label must be a boolean or numeric 0/1")


def sample_to_swift_row(sample: Sample, stage: str) -> dict:
    validate_sample_for_stage(sample, stage)
    extras = _sample_extras(sample)

    if stage == "sft":
        return {
            "id": sample.id,
            "messages": _to_messages(sample.messages),
            **extras,
        }
    if stage == "dpo":
        messages = _to_messages(sample.prompt) + _chosen_messages(sample.chosen)
        return {
            "id": sample.id,
            "messages": messages,
            "rejected_response": _rejected_response(sample.rejected),
            **extras,
        }
    if stage == "kto":
        messages = _to_messages(sample.prompt)
        if sample.completion is not None:
            messages.append({"role": "assistant", "content": sample.completion})
        return {
            "id": sample.id,
            "messages": messages,
            "label": _kto_label(sample.label),
            **extras,
        }
    raise IRValidationError(f"swift adapter does not support stage={stage!r}")


def write_swift_jsonl(stream: Iterable[Sample], recipe, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in stream:
            row = sample_to_swift_row(sample, recipe.stage)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    if n == 0:
        raise IRValidationError("no samples produced - empty SampleStream")
    return out_path
