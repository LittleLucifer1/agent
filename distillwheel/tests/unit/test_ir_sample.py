import pytest

from distillwheel.core.errors import IRValidationError
from distillwheel.core.ir.sample import Message, Sample, iter_samples_from_jsonl, validated_sample_stream


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


def test_sft_requires_final_nonempty_assistant():
    ends_with_user = Sample(
        id="x3",
        task_type="sft",
        messages=[Message(role="assistant", content="a"), Message(role="user", content="q")],
    )
    with pytest.raises(IRValidationError, match="last message"):
        ends_with_user.validate()

    empty_answer = Sample(
        id="x4",
        task_type="sft",
        messages=[Message(role="user", content="q"), Message(role="assistant", content="   ")],
    )
    with pytest.raises(IRValidationError, match="non-empty final"):
        empty_answer.validate()


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


@pytest.mark.parametrize("field", ["prompt", "chosen", "rejected"])
def test_preference_fields_must_be_nonempty(field):
    values = {"prompt": "q", "chosen": "a", "rejected": "b"}
    values[field] = "  "
    s = Sample(id="p3", task_type="preference", **values)
    with pytest.raises(IRValidationError, match=field):
        s.validate()


def test_message_or_text_lists_validate_message_roles():
    s = Sample(
        id="p-role",
        task_type="preference",
        prompt=[Message(role="invalid", content="q")],  # type: ignore[arg-type]
        chosen=[Message(role="assistant", content="a")],
        rejected=[Message(role="assistant", content="b")],
    )
    with pytest.raises(IRValidationError, match="role.*invalid"):
        s.validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("images", [""], "images"),
        ("images", "image.png", "images"),
        ("tools", ["not-a-mapping"], "tools"),
    ],
)
def test_sample_validates_multimodal_and_tool_container_types(field, value, message):
    sample = Sample(
        id="s-assets",
        task_type="sft",
        messages=[Message("user", "hi"), Message("assistant", "hello")],
    )
    setattr(sample, field, value)
    with pytest.raises(IRValidationError, match=message):
        sample.validate()


def test_message_validates_tool_call_fields():
    sample = Sample(
        id="s-tool-call",
        task_type="sft",
        messages=[
            Message("user", "hi", tool_calls=["not-a-mapping"]),
            Message("assistant", "hello"),
        ],
    )
    with pytest.raises(IRValidationError, match="tool_calls"):
        sample.validate()


def test_sft_allows_final_assistant_tool_call_without_text():
    sample = Sample(
        id="s-final-tool",
        task_type="sft",
        messages=[
            Message("user", "weather?"),
            Message(
                "assistant",
                "",
                tool_calls=[{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": "{}"},
                }],
            ),
        ],
    )
    sample.validate()


def test_tool_call_fields_are_role_scoped():
    user_call = Sample(
        id="s-user-call",
        task_type="sft",
        messages=[
            Message("user", "q", tool_calls=[{"id": "call-1"}]),
            Message("assistant", "a"),
        ],
    )
    with pytest.raises(IRValidationError, match="only valid for assistant"):
        user_call.validate()

    missing_id = Sample(
        id="s-tool-no-id",
        task_type="sft",
        messages=[Message("tool", "result"), Message("assistant", "answer")],
    )
    with pytest.raises(IRValidationError, match="require `tool_call_id`"):
        missing_id.validate()


def test_kto_requires_label():
    s = Sample(id="k1", task_type="kto", prompt="q", completion="a")
    with pytest.raises(IRValidationError):
        s.validate()


def test_kto_ok():
    s = Sample(id="k2", task_type="kto", prompt="q", completion="a", label=True)
    s.validate()


def test_kto_rejects_non_binary_numeric_label():
    s = Sample(id="k2b", task_type="kto", prompt="q", completion="a", label=2)
    with pytest.raises(IRValidationError, match="bool, 0, or 1"):
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


def test_sample_from_dict_rejects_unknown_fields():
    raw = Sample(
        id="strict", task_type="sft",
        messages=[Message("user", "q"), Message("assistant", "a")],
    ).to_dict()
    raw["image"] = "typo.png"
    with pytest.raises(IRValidationError, match="unknown sample field"):
        Sample.from_dict(raw)


def test_message_from_dict_rejects_unknown_fields():
    with pytest.raises(IRValidationError, match="unknown message field"):
        Message.from_dict({"role": "user", "content": "q", "contents": "typo"})


def test_jsonl_error_has_path_and_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("\n{not-json}\n", encoding="utf-8")
    with pytest.raises(IRValidationError, match=r"bad\.jsonl:2"):
        list(iter_samples_from_jsonl(path))


def test_jsonl_invalid_utf8_is_wrapped(tmp_path):
    path = tmp_path / "bad-utf8.jsonl"
    path.write_bytes(b"\xff\n")
    with pytest.raises(IRValidationError, match="bad-utf8.jsonl"):
        list(iter_samples_from_jsonl(path))


def test_validated_stream_enforces_recipe_stage():
    preference = Sample(
        id="p4", task_type="preference", prompt="q", chosen="a", rejected="b"
    )
    with pytest.raises(IRValidationError, match="incompatible"):
        list(validated_sample_stream([preference], "sft"))
