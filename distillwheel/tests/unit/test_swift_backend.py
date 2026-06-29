import json
from pathlib import Path

import pytest

from distillwheel.backends.swift.data_writer import sample_to_swift_row, write_swift_jsonl
from distillwheel.backends.swift.recipe_mapper import recipe_to_swift_args
from distillwheel.core.ir.recipe import (
    IOConfig, OptimConfig, PEFTConfig, Recipe, TrainConfig,
)
from distillwheel.core.ir.sample import Message, Sample


def _sft_sample(i):
    return Sample(
        id=f"s{i}",
        task_type="sft",
        messages=[
            Message(role="user", content="q"),
            Message(role="assistant", content="a"),
        ],
    )


def test_swift_sft_row():
    row = sample_to_swift_row(_sft_sample(1), "sft")
    assert row["messages"][0]["role"] == "user"


def test_swift_dpo_row():
    s = Sample(id="d1", task_type="preference",
               prompt="q", chosen="good", rejected="bad")
    row = sample_to_swift_row(s, "dpo")
    assert row["chosen_response"] == "good"
    assert row["rejected_response"] == "bad"
    assert row["messages"][0]["content"] == "q"


def test_swift_kto_row():
    s = Sample(id="k1", task_type="kto", prompt="q", completion="a", label=True)
    row = sample_to_swift_row(s, "kto")
    assert row["label"] is True
    assert row["messages"][-1]["role"] == "assistant"


def test_swift_write_jsonl(tmp_path):
    out = tmp_path / "data.jsonl"
    recipe = Recipe(
        stage="sft", base_model="m",
        train=TrainConfig(), optim=OptimConfig(),
        io=IOConfig(output_dir=str(tmp_path)),
    )
    write_swift_jsonl((_sft_sample(i) for i in range(3)), recipe, out)
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert rows[0]["id"] == "s0"


def test_swift_recipe_mapper_writes_yaml(tmp_path):
    recipe = Recipe(
        stage="sft", base_model="Qwen/Qwen2-0.5B",
        train=TrainConfig(epochs=2, micro_batch=2, grad_accum=4, max_len=2048),
        optim=OptimConfig(lr=1e-5),
        io=IOConfig(output_dir=str(tmp_path)),
        peft=PEFTConfig(type="lora", r=8, alpha=16, target_modules=["q_proj", "v_proj"]),
        target_template="qwen",
    )
    data = tmp_path / "data.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    cfg = recipe_to_swift_args(recipe, data, tmp_path / "swift.yaml")
    import yaml

    parsed = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert parsed["model"] == "Qwen/Qwen2-0.5B"
    assert parsed["sft_type"] == "lora"
    assert parsed["lora_target_modules"] == ["q_proj", "v_proj"]
    assert parsed["template"] == "qwen"
    assert parsed["__subcommand__"] == "sft"


def test_swift_recipe_mapper_dpo_rlhf(tmp_path):
    recipe = Recipe(
        stage="dpo", base_model="m",
        train=TrainConfig(), optim=OptimConfig(),
        io=IOConfig(output_dir=str(tmp_path)),
    )
    data = tmp_path / "data.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    cfg = recipe_to_swift_args(recipe, data, tmp_path / "swift.yaml")
    import yaml

    parsed = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert parsed["__subcommand__"] == "rlhf"
    assert parsed["rlhf_type"] == "dpo"
