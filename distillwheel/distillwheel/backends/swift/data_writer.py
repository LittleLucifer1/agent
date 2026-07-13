"""Translate Sample IR rows into ms-swift 4.4 JSONL records.

The preference schema follows ms-swift's standard form: the chosen answer is
the final assistant message in ``messages`` and ``rejected_response`` is the
only additional response field.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable, List

from ...core.errors import IRValidationError
from ...core.ir.sample import Message, Sample


def _message_dict(value) -> dict:
    if isinstance(value, Message):
        return value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    raise IRValidationError(
        f"expected Message or mapping, got {type(value).__name__}"
    )


def _to_messages(value) -> List[dict]:
    if value is None:
        return []
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, (Message, Mapping)):
        return [_message_dict(value)]
    return [_message_dict(message) for message in value]


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


def _response_text(value) -> str:
    """Extract the textual rejected response required by ms-swift."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value

    messages = _to_messages(value)
    assistant_parts = [
        message.get("content", "")
        for message in messages
        if message.get("role") == "assistant"
    ]
    parts = assistant_parts or [message.get("content", "") for message in messages]
    if not all(isinstance(part, str) for part in parts):
        raise IRValidationError("DPO rejected_response must be representable as text")
    return "\n".join(parts)


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
    sample.validate()
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
            "rejected_response": _response_text(sample.rejected),
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
