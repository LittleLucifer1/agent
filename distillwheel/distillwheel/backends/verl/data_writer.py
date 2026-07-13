"""Stream DistillWheel RL prompts to VERL 0.8-compatible parquet.

VERL's :class:`RLHFDataset` expects ``prompt`` to be a nested Arrow value
(``list<struct<role, content>>``), not a JSON string.  It also consumes the
``data_source``, ``ability``, ``reward_model`` and ``extra_info`` columns
directly, so those columns are written as native Arrow values as well.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional

from ...core.errors import IRValidationError
from ...core.ir.sample import Message, Sample

_BATCH = 1024
_ROLES = {"system", "user", "assistant", "tool"}


def _text(value: Any, *, field: str, sample_id: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise IRValidationError(
            f"sample {sample_id!r}: {field} must be a string, got {type(value).__name__}"
        )
    if not allow_empty and not value.strip():
        raise IRValidationError(f"sample {sample_id!r}: {field} must not be empty")
    return value


def _prompt_messages(prompt: Any, sample_id: str) -> List[dict]:
    """Return VERL chat messages while validating the nested prompt shape."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": _text(prompt, field="prompt", sample_id=sample_id)}]
    if not isinstance(prompt, (list, tuple)) or not prompt:
        raise IRValidationError(
            f"sample {sample_id!r}: verl requires a non-empty string or message-list prompt"
        )

    messages: List[dict] = []
    for index, item in enumerate(prompt):
        if isinstance(item, Message):
            role, content = item.role, item.content
            if item.tool_calls is not None or item.tool_call_id is not None:
                raise IRValidationError(
                    f"sample {sample_id!r}: VERL 0.8 adapter does not yet map "
                    "Message tool-call fields"
                )
        elif isinstance(item, Mapping):
            role, content = item.get("role"), item.get("content")
        else:
            raise IRValidationError(
                f"sample {sample_id!r}: prompt[{index}] must be Message or mapping"
            )
        if role not in _ROLES:
            raise IRValidationError(
                f"sample {sample_id!r}: prompt[{index}].role={role!r} is unsupported"
            )
        if role == "tool":
            raise IRValidationError(
                f"sample {sample_id!r}: VERL 0.8 tool rollouts require a "
                "tool-config schema that the current Recipe IR cannot express"
            )
        messages.append({
            "role": role,
            "content": _text(
                content,
                field=f"prompt[{index}].content",
                sample_id=sample_id,
            ),
        })
    return messages


def _scalar_text(value: Any, *, field: str, sample_id: str, default: str = "") -> str:
    """Coerce scalar metadata without serialising a nested object as JSON."""
    if value is None:
        return default
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    raise IRValidationError(
        f"sample {sample_id!r}: {field} must be a scalar, got {type(value).__name__}"
    )


def _reward_model(meta: Mapping[str, Any], sample_id: str) -> dict:
    raw = meta.get("reward_model") or {}
    if not isinstance(raw, Mapping):
        raise IRValidationError(f"sample {sample_id!r}: meta.reward_model must be a mapping")
    ground_truth = raw.get("ground_truth", meta.get("ground_truth", meta.get("answer")))
    return {
        "style": _scalar_text(
            raw.get("style", meta.get("reward_style", "rule")),
            field="meta.reward_model.style",
            sample_id=sample_id,
            default="rule",
        ),
        "ground_truth": _scalar_text(
            ground_truth,
            field="meta.reward_model.ground_truth",
            sample_id=sample_id,
        ),
    }


def _extra_info(meta: Mapping[str, Any], sample_id: str, row_index: int) -> dict:
    raw = meta.get("extra_info") or {}
    if not isinstance(raw, Mapping):
        raise IRValidationError(f"sample {sample_id!r}: meta.extra_info must be a mapping")

    # Common VERL reward-function fields remain directly addressable with
    # ``extra_info.get(...)``.  Other scalar metadata is preserved as a nested
    # key/value list rather than collapsing the complete object into JSON.
    def pick(key: str, default: Any = "") -> Any:
        return raw[key] if key in raw else meta.get(key, default)

    reserved = {
        "data_source", "ability", "reward_model", "ground_truth", "reward_style",
        "extra_info", "index", "sample_id", "split", "question", "answer",
    }
    metadata = []
    merged = {k: v for k, v in meta.items() if k not in reserved}
    merged.update({k: v for k, v in raw.items() if k not in reserved})
    for key in sorted(merged, key=str):
        value = merged[key]
        if value is None or isinstance(value, (str, int, float, bool)):
            metadata.append({"key": str(key), "value": "" if value is None else str(value)})

    index_value = pick("index", row_index)
    if isinstance(index_value, bool) or not isinstance(index_value, int):
        raise IRValidationError(f"sample {sample_id!r}: meta.extra_info.index must be an integer")
    return {
        "index": index_value,
        "sample_id": _scalar_text(pick("sample_id", sample_id), field="extra_info.sample_id", sample_id=sample_id),
        "split": _scalar_text(pick("split"), field="extra_info.split", sample_id=sample_id),
        "question": _scalar_text(pick("question"), field="extra_info.question", sample_id=sample_id),
        "answer": _scalar_text(pick("answer", meta.get("ground_truth", "")), field="extra_info.answer", sample_id=sample_id),
        "metadata": metadata,
    }


def write_verl_parquet(stream: Iterable[Sample], recipe: Any, out_path: Path) -> Path:
    """Write only ``rl_prompt`` samples using VERL's native nested schema."""
    del recipe  # The dataset representation is stage-independent for VERL RL.
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise RuntimeError(
            "pyarrow is required by the verl adapter; install distillwheel[verl]"
        ) from exc

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    message_type = pa.struct([("role", pa.string()), ("content", pa.string())])
    reward_type = pa.struct([("style", pa.string()), ("ground_truth", pa.string())])
    extra_type = pa.struct([
        ("index", pa.int64()),
        ("sample_id", pa.string()),
        ("split", pa.string()),
        ("question", pa.string()),
        ("answer", pa.string()),
        ("metadata", pa.list_(pa.struct([("key", pa.string()), ("value", pa.string())]))),
    ])
    schema = pa.schema([
        ("data_source", pa.string()),
        ("prompt", pa.list_(message_type)),
        ("ability", pa.string()),
        ("reward_model", reward_type),
        ("extra_info", extra_type),
    ])

    writer: Optional[pq.ParquetWriter] = None
    columns = {name: [] for name in schema.names}
    count = 0
    try:
        for row_index, sample in enumerate(stream):
            if not isinstance(sample, Sample):
                raise IRValidationError(
                    f"verl data stream yielded {type(sample).__name__}, expected Sample"
                )
            if sample.task_type != "rl_prompt":
                raise IRValidationError(
                    f"sample {sample.id!r}: verl accepts only task_type='rl_prompt', "
                    f"got {sample.task_type!r}"
                )
            if not isinstance(sample.id, str) or not sample.id.strip():
                raise IRValidationError("verl requires every sample to have a non-empty string id")
            if not isinstance(sample.meta, Mapping):
                raise IRValidationError(f"sample {sample.id!r}: meta must be a mapping")
            if sample.images:
                raise IRValidationError(
                    f"sample {sample.id!r}: VERL multimodal prompts are not yet "
                    "mapped by this adapter; refusing to drop images"
                )
            if sample.tools:
                raise IRValidationError(
                    f"sample {sample.id!r}: VERL tool definitions are not yet "
                    "mapped by this adapter; refusing to drop tools"
                )

            meta = sample.meta
            columns["data_source"].append(_scalar_text(
                meta.get("data_source", "distillwheel/custom"),
                field="meta.data_source", sample_id=sample.id,
            ))
            columns["prompt"].append(_prompt_messages(sample.prompt, sample.id))
            columns["ability"].append(_scalar_text(
                meta.get("ability", "general"), field="meta.ability", sample_id=sample.id,
            ))
            columns["reward_model"].append(_reward_model(meta, sample.id))
            columns["extra_info"].append(_extra_info(meta, sample.id, row_index))
            count += 1

            if count % _BATCH == 0:
                writer = writer or pq.ParquetWriter(out_path, schema)
                _flush(writer, schema, columns)

        if count == 0:
            raise IRValidationError("verl data stream is empty; at least one rl_prompt sample is required")
        if columns["prompt"]:
            writer = writer or pq.ParquetWriter(out_path, schema)
            _flush(writer, schema, columns)
    finally:
        if writer is not None:
            writer.close()

    return out_path


def _flush(writer: Any, schema: Any, columns: dict) -> None:
    import pyarrow as pa

    writer.write_table(pa.table(columns, schema=schema))
    for values in columns.values():
        values.clear()
