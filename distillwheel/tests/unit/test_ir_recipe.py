from pathlib import Path

import pytest

from distillwheel.core.errors import IRValidationError
from distillwheel.core.ir.recipe import (
    IOConfig,
    OptimConfig,
    PEFTConfig,
    RECIPE_SCHEMA_VERSION,
    Recipe,
    RLConfig,
    TrainConfig,
)


def _basic_sft_recipe(out_dir: Path) -> Recipe:
    return Recipe(
        stage="sft",
        base_model="Qwen/Qwen2-0.5B",
        train=TrainConfig(epochs=1, global_batch=8, micro_batch=2, grad_accum=4),
        optim=OptimConfig(lr=5e-5),
        io=IOConfig(output_dir=str(out_dir)),
        peft=PEFTConfig(type="lora"),
        target_template="qwen",
    )


def test_recipe_yaml_roundtrip(tmp_path):
    r = _basic_sft_recipe(tmp_path / "out")
    p = tmp_path / "r.yaml"
    r.to_yaml(p)
    r2 = Recipe.from_yaml(p)
    assert r2.stage == "sft"
    assert r2.train.grad_accum == 4
    assert r2.peft is not None and r2.peft.type == "lora"


def test_recipe_validate_rl_required():
    r = Recipe(
        stage="grpo",
        base_model="m",
        train=TrainConfig(),
        optim=OptimConfig(),
        io=IOConfig(output_dir="o"),
    )
    with pytest.raises(IRValidationError):
        r.validate()


def test_recipe_validate_batch_invariant():
    r = Recipe(
        stage="sft",
        base_model="m",
        train=TrainConfig(global_batch=1, micro_batch=4),
        optim=OptimConfig(),
        io=IOConfig(output_dir="o"),
    )
    with pytest.raises(IRValidationError):
        r.validate()


def test_recipe_schema_version_check(tmp_path):
    r = _basic_sft_recipe(tmp_path / "out")
    p = tmp_path / "r.yaml"
    r.to_yaml(p)
    text = p.read_text(encoding="utf-8")
    text = text.replace(
        f"_recipe_schema_version: {RECIPE_SCHEMA_VERSION}",
        "_recipe_schema_version: 999",
    )
    p.write_text(text, encoding="utf-8")
    with pytest.raises(IRValidationError):
        Recipe.from_yaml(p)
