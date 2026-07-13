"""Sample / Message IR — framework-neutral training-data representation.

The IR layer stores **raw text only**: no tokenization, no chat-template
application. Each backend's ``prepare_data`` is responsible for translating
the IR into its native format (JSONL, parquet, ...).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Literal, Optional, Protocol, Union, runtime_checkable

from ..errors import IRValidationError

TaskType = Literal["sft", "preference", "kto", "rl_prompt"]
Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: str
    tool_calls: Optional[List[dict]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"role": self.role, "content": self.content}
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        return cls(
            role=d["role"],
            content=d.get("content", ""),
            tool_calls=d.get("tool_calls"),
            tool_call_id=d.get("tool_call_id"),
        )


MessageOrText = Union[str, List[Message], None]


@dataclass
class Sample:
    """Framework-neutral training sample.

    Only the fields relevant to ``task_type`` need to be populated;
    :meth:`validate` enforces the per-task field requirements.
    """

    id: str
    task_type: TaskType
    messages: Optional[List[Message]] = None
    prompt: MessageOrText = None
    chosen: MessageOrText = None
    rejected: MessageOrText = None
    completion: Optional[str] = None
    label: Union[bool, float, None] = None
    tools: Optional[List[dict]] = None
    images: Optional[List[str]] = None
    meta: dict = field(default_factory=dict)

    def validate(self) -> None:
        """Validate field combinations against ``task_type``.

        Heavy lifting lives in :mod:`distillwheel.core.ir.validators` to
        avoid an import cycle with the validators module itself.
        """
        from .validators import validate_sample

        validate_sample(self)

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.messages is not None:
            d["messages"] = [m.to_dict() for m in self.messages]
        for f in ("prompt", "chosen", "rejected"):
            v = getattr(self, f)
            if isinstance(v, list):
                d[f] = [m.to_dict() if isinstance(m, Message) else m for m in v]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        def _coerce_messages(v):
            if v is None or isinstance(v, str):
                return v
            if isinstance(v, list):
                return [Message.from_dict(m) if isinstance(m, dict) else m for m in v]
            return v

        return cls(
            id=d["id"],
            task_type=d["task_type"],
            messages=[Message.from_dict(m) for m in d["messages"]] if d.get("messages") else None,
            prompt=_coerce_messages(d.get("prompt")),
            chosen=_coerce_messages(d.get("chosen")),
            rejected=_coerce_messages(d.get("rejected")),
            completion=d.get("completion"),
            label=d.get("label"),
            tools=d.get("tools"),
            images=d.get("images"),
            meta=d.get("meta", {}),
        )


@runtime_checkable
class SampleStream(Protocol):
    """Anything iterable of :class:`Sample` qualifies.

    Adapters consume ``SampleStream`` lazily so a multi-million-row dataset
    can be written to parquet/JSONL without loading the whole list.
    """

    def __iter__(self) -> Iterator[Sample]:  # pragma: no cover - protocol
        ...


def iter_samples_from_jsonl(path: str | Path) -> Iterable[Sample]:
    """Yield validated samples and attach file/line context to input errors."""
    import json

    source = Path(path)
    try:
        with open(source, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise IRValidationError("sample JSON value must be an object")
                    sample = Sample.from_dict(raw)
                    sample.validate()
                except (json.JSONDecodeError, KeyError, TypeError, ValueError, IRValidationError) as exc:
                    raise IRValidationError(
                        f"invalid sample at {source}:{line_number}: {exc}"
                    ) from exc
                yield sample
    except OSError as exc:
        raise IRValidationError(f"cannot read sample JSONL {source}: {exc}") from exc


def validated_sample_stream(stream: Iterable[Sample], stage: str) -> Iterator[Sample]:
    """Lazily enforce both the Sample schema and Recipe-stage compatibility."""
    from .validators import validate_sample_for_stage

    for index, sample in enumerate(stream, start=1):
        if not isinstance(sample, Sample):
            raise IRValidationError(
                f"sample stream item {index} must be Sample, got {type(sample).__name__}"
            )
        try:
            validate_sample_for_stage(sample, stage)
        except IRValidationError as exc:
            raise IRValidationError(f"sample stream item {index}: {exc}") from exc
        yield sample
