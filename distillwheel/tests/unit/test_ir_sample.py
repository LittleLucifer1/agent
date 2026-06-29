import pytest

from distillwheel.core.errors import IRValidationError
from distillwheel.core.ir.sample import Message, Sample


def test_sft_sample_valid():
    s = Sample(
        id="x1",
        task_type="sft",
        messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ],
    )
    s.validate()


def test_sft_requires_assistant():
    s = Sample(
        id="x2",
        task_type="sft",
        messages=[Message(role="user", content="hi")],
    )
    with pytest.raises(IRValidationError):
        s.validate()


def test_preference_requires_three_fields():
    s = Sample(id="p1", task_type="preference", prompt="q", chosen="a")
    with pytest.raises(IRValidationError):
        s.validate()


def test_preference_ok():
    s = Sample(
        id="p2", task_type="preference",
        prompt="q", chosen="a", rejected="b",
    )
    s.validate()


def test_kto_requires_label():
    s = Sample(id="k1", task_type="kto", prompt="q", completion="a")
    with pytest.raises(IRValidationError):
        s.validate()


def test_kto_ok():
    s = Sample(id="k2", task_type="kto", prompt="q", completion="a", label=True)
    s.validate()


def test_rl_prompt_ok():
    s = Sample(id="r1", task_type="rl_prompt", prompt="solve this")
    s.validate()


def test_unknown_task_type():
    s = Sample(id="u1", task_type="weird")  # type: ignore[arg-type]
    with pytest.raises(IRValidationError):
        s.validate()


def test_sample_roundtrip():
    s = Sample(
        id="r2", task_type="sft",
        messages=[Message(role="user", content="hi"),
                  Message(role="assistant", content="hello")],
        meta={"src": "test"},
    )
    s2 = Sample.from_dict(s.to_dict())
    assert s2.id == s.id
    assert s2.task_type == s.task_type
    assert s2.messages[1].content == "hello"
    assert s2.meta == {"src": "test"}
