"""Sample / Message IR — framework-neutral training-data representation.

The IR layer stores **raw text only**: no tokenization, no chat-template
application. Each backend's ``prepare_data`` is responsible for translating
the IR into its native format (JSONL, parquet, ...).

Field usage by task_type
========================

Each task_type only requires a subset of Sample fields. The table below
shows which fields are **required (R)**, **optional (O)**, or **unused (-)** :

+-------------+----------+--------+--------+----------+------------+-------+-------+--------+
| field       | sft      | preference (DPO) | kto      | rl_prompt  | notes                  |
+=============+==========+==================+==========+============+========================+
| id          |  R       |  R               |  R       |  R         | 全局唯一标识            |
| task_type   |  R       |  R               |  R       |  R         | 决定校验规则和路由      |
| messages    |  R       |  -               |  -       |  -         | 完整多轮对话            |
| prompt      |  -       |  R               |  R       |  R         | 用户问题/提示           |
| chosen      |  -       |  R               |  -       |  -         | 偏好学习的好回答        |
| rejected    |  -       |  R               |  -       |  -         | 偏好学习的差回答        |
| completion  |  -       |  -               |  R       |  -         | 单条回答(供KTO打分)     |
| label       |  -       |  -               |  R       |  -         | True=好/False=差        |
| tools       |  O       |  -               |  -       |  -         | function calling 定义   |
| images      |  O       |  O               |  O       |  O         | 多模态(v1保留未实现)    |
| meta        |  O       |  O               |  O       |  O         | 透传业务自定义字段      |
+-------------+----------+------------------+----------+------------+------------------------+

Examples
========
**SFT** — supervised fine-tuning, 完整多轮对话, 最后一条 assistant 消息就是要学的目标:
    Sample(
        id="sft_001",
        task_type="sft",
        messages=[
            Message(role="system",    content="你是一个数学助手"),
            Message(role="user",      content="1+1等于几？"),
            Message(role="assistant", content="1+1等于2。"),
        ],
    )

**SFT + tool calling** — 教模型学会调用函数:
    Sample(
        id="sft_tool_001",
        task_type="sft",
        messages=[
            Message(role="user",      content="北京今天天气怎么样？"),
            Message(role="assistant", content="",
                    tool_calls=[{"id": "call_1", "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city":"北京"}'}}]),
            Message(role="tool",      content='{"temp":28,"desc":"晴"}',
                    tool_call_id="call_1"),
            Message(role="assistant", content="北京今天28°C，晴天。"),
        ],
        tools=[{"type": "function",
                "function": {"name": "get_weather",
                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
    )

**Preference / DPO** — 给同一个 prompt 提供好/坏两个回答，训练偏好:
    Sample(
        id="dpo_001",
        task_type="preference",
        prompt="请用一句话解释量子纠缠",                     # 也可以是 list[Message]
        chosen="量子纠缠是两个粒子无论距离多远都能瞬间相互关联的现象。",
        rejected="量子纠缠就是两个东西绑在一起。",
    )

**KTO** — 单条回答 + 二元标签(好/差), 不需要配对:
    Sample(
        id="kto_001",
        task_type="kto",
        prompt="写一首关于春天的诗",
        completion="春风又绿江南岸，明月何时照我还。",
        label=True,   # True=好回答, False=差回答
    )

**RL / GRPO** — 只有 prompt, 模型自己 rollout 生成回答, reward function 打分:
    Sample(
        id="rl_001",
        task_type="rl_prompt",
        prompt="求解方程 2x + 3 = 7",                       # 也可以是 list[Message]
    )
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator, List, Literal, Optional, Protocol, Union, runtime_checkable

TaskType = Literal["sft", "preference", "kto", "rl_prompt"]
Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    """对话中的单条消息。"""
    role: Role                                   # system / user / assistant / tool
    content: str                                 # 消息正文
    tool_calls: Optional[List[dict]] = None      # assistant 发起的函数调用 (SFT tool-calling)
    tool_call_id: Optional[str] = None           # tool 消息关联的 call id

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

    # ── 通用字段 (所有 task_type 都需要) ──────────────────────────────
    id: str                                      # 样本唯一标识
    task_type: TaskType                          # 决定哪些字段必填, 以及数据如何被消费

    # ── SFT 字段 ─────────────────────────────────────────────────────
    # 完整多轮对话, 最后一条 assistant 消息 = 训练目标
    messages: Optional[List[Message]] = None

    # ── Preference (DPO) + KTO + RL 共用 ────────────────────────────
    # 用户输入/提示; DPO 和 KTO 用它做条件, RL 用它做 rollout 起点
    prompt: MessageOrText = None

    # ── Preference (DPO) 专用 ────────────────────────────────────────
    # 同一 prompt 下的好回答和差回答, 成对出现
    chosen: MessageOrText = None                 # 偏好学习: 好的回答
    rejected: MessageOrText = None               # 偏好学习: 差的回答

    # ── KTO 专用 ─────────────────────────────────────────────────────
    # 单条回答 + 二元标签, 不需要 chosen/rejected 配对
    completion: Optional[str] = None             # 待评价的回答
    label: Union[bool, float, None] = None       # True/1.0=好, False/0.0=差

    # ── 可选扩展 (所有 task_type 均可携带) ───────────────────────────
    tools: Optional[List[dict]] = None           # function calling 的工具定义 (主要用于 SFT)
    images: Optional[List[str]] = None           # 多模态图片路径/URL (v1 保留, 暂未实现)
    meta: dict = field(default_factory=dict)     # 业务自定义透传字段, 不影响训练逻辑

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


def iter_samples_from_jsonl(path: str) -> Iterable[Sample]:
    """Convenience: yield :class:`Sample` from a JSONL file."""
    import json

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield Sample.from_dict(json.loads(line))
