"""IR → swift JSONL writer.

swift expects different field shapes per task:

* SFT: ``{"messages": [...]}``
* DPO: ``{"messages": [..prompt..], "rejected_response": "..."}``
  (swift also accepts ``chosen`` / ``rejected`` as text fields)
* KTO: ``{"messages": [...], "label": true/false}``

We translate at the data-writer level so the recipe_mapper just points
at a single file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from ...core.ir.sample import Message, Sample


def _to_messages(value) -> List[dict]:
    if value is None:
        return []
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    return [m.to_dict() if isinstance(m, Message) else m for m in value]


def _flatten_text(value) -> str:
    """Best-effort text extraction for response-style fields."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    parts = []
    for m in value:
        if isinstance(m, Message):
            parts.append(m.content)
        elif isinstance(m, dict):
            parts.append(m.get("content", ""))
    return "\n".join(parts)


def sample_to_swift_row(sample: Sample, stage: str) -> dict:
    sample.validate()
    if stage == "sft":
        return {
            "id": sample.id,
            "messages": [m.to_dict() for m in (sample.messages or [])],
            **({"tools": sample.tools} if sample.tools else {}),
        }
    if stage == "dpo":
        return {
            "id": sample.id,
            "messages": _to_messages(sample.prompt),
            "chosen_response": _flatten_text(sample.chosen),
            "rejected_response": _flatten_text(sample.rejected),
        }
    if stage == "kto":
        msgs = _to_messages(sample.prompt)
        if sample.completion:
            msgs = msgs + [{"role": "assistant", "content": sample.completion}]
        return {
            "id": sample.id,
            "messages": msgs,
            "label": bool(sample.label),
        }
    raise ValueError(f"swift adapter does not support stage={stage!r}")


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
        raise ValueError("no samples produced — empty SampleStream")
    return out_path
