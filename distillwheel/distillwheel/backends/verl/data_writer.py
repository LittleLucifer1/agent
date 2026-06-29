"""IR → verl parquet writer.

verl's PPO/GRPO trainer reads prompts from a parquet file with at least
one ``prompt`` column. We stream-write with ``pyarrow.ParquetWriter`` so
multi-million-row prompt pools don't have to live in RAM.

For prompt-style stages (grpo/ppo/rloo/opd), each Sample is expected to
have ``sample.prompt`` populated (string or list[Message]).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

from ...core.errors import IRValidationError
from ...core.ir.sample import Message, Sample

_BATCH = 1024


def _prompt_text(prompt) -> str:
    if prompt is None:
        raise IRValidationError("verl backend requires non-empty `prompt`")
    if isinstance(prompt, str):
        return prompt
    parts = []
    for m in prompt:
        if isinstance(m, Message):
            parts.append(f"{m.role}: {m.content}")
        elif isinstance(m, dict):
            parts.append(f"{m.get('role','user')}: {m.get('content','')}")
    return "\n".join(parts)


def _prompt_messages(prompt) -> List[dict]:
    if prompt is None:
        return []
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    out = []
    for m in prompt:
        if isinstance(m, Message):
            out.append(m.to_dict())
        elif isinstance(m, dict):
            out.append(m)
    return out


def write_verl_parquet(stream: Iterable[Sample], recipe, out_path: Path) -> Path:
    """Stream-write a parquet file with columns:
       id (string), prompt (string), messages (json-string), meta (json-string).
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required by the verl adapter. "
            "Install distillwheel[verl] or pip install pyarrow."
        ) from e

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([
        ("id", pa.string()),
        ("prompt", pa.string()),
        ("messages", pa.string()),  # json-encoded
        ("meta", pa.string()),
    ])

    writer: Optional[pq.ParquetWriter] = None
    buf_id: List[str] = []
    buf_prompt: List[str] = []
    buf_msgs: List[str] = []
    buf_meta: List[str] = []
    n = 0
    try:
        for sample in stream:
            if sample.task_type not in ("rl_prompt", "sft"):
                # SFT samples are allowed: their messages encode the prompt;
                # other types (preference/kto) don't make sense for verl RL.
                if sample.task_type != "rl_prompt":
                    # try to derive a prompt from messages
                    if not sample.messages:
                        raise IRValidationError(
                            f"sample {sample.id!r}: verl needs `prompt` or `messages`"
                        )
            prompt_value = sample.prompt if sample.prompt is not None else sample.messages
            buf_id.append(sample.id)
            buf_prompt.append(_prompt_text(prompt_value))
            buf_msgs.append(json.dumps(_prompt_messages(prompt_value), ensure_ascii=False))
            buf_meta.append(json.dumps(sample.meta or {}, ensure_ascii=False))
            n += 1
            if n % _BATCH == 0:
                if writer is None:
                    writer = pq.ParquetWriter(out_path, schema)
                _flush(writer, schema, buf_id, buf_prompt, buf_msgs, buf_meta)
                buf_id.clear(); buf_prompt.clear(); buf_msgs.clear(); buf_meta.clear()

        if buf_id:
            if writer is None:
                writer = pq.ParquetWriter(out_path, schema)
            _flush(writer, schema, buf_id, buf_prompt, buf_msgs, buf_meta)
    finally:
        if writer is not None:
            writer.close()

    if n == 0:
        raise IRValidationError("no samples produced — empty SampleStream")
    return out_path


def _flush(writer, schema, ids, prompts, msgs, metas):
    import pyarrow as pa
    table = pa.table({
        "id": ids,
        "prompt": prompts,
        "messages": msgs,
        "meta": metas,
    }, schema=schema)
    writer.write_table(table)
