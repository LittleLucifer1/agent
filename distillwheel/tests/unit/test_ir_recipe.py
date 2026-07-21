from pathlib import Path
from typing import get_type_hints

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


@pytest.mark.parametrize("version", [True, 1.0, "1"])
def test_recipe_schema_version_requires_an_exact_integer(version):
    raw = _basic_sft_recipe(Path("out")).to_dict()
    raw["_recipe_schema_version"] = version
    with pytest.raises(IRValidationError, match="unsupported recipe schema version"):
        Recipe.from_dict(raw)


def test_default_batch_is_internally_consistent():
    train = TrainConfig()
    assert train.global_batch == train.micro_batch * train.grad_accum
    assert train.grad_accum == 32


def test_recipe_from_dict_does_not_mutate_input(tmp_path):
    recipe = _basic_sft_recipe(tmp_path / "out")
    raw = recipe.to_dict()
    before = dict(raw)
    Recipe.from_dict(raw)
    assert raw == before


def test_recipe_rejects_unknown_stage():
    recipe = Recipe(
        stage="opd",  # type: ignore[arg-type]
        base_model="m",
        train=TrainConfig(),
        optim=OptimConfig(),
        io=IOConfig(output_dir="o"),
        rl=RLConfig(),
    )
    with pytest.raises(IRValidationError, match="unknown recipe.stage"):
        recipe.validate()


def test_recipe_rejects_inconsistent_effective_batch():
    recipe = Recipe(
        stage="sft",
        base_model="m",
        train=TrainConfig(global_batch=16, micro_batch=1, grad_accum=8),
        optim=OptimConfig(),
        io=IOConfig(output_dir="o"),
    )
    with pytest.raises(IRValidationError, match="must equal"):
        recipe.validate()


@pytest.mark.parametrize(
    "train",
    [
        TrainConfig(global_batch=0),
        TrainConfig(micro_batch=0),
        TrainConfig(grad_accum=0),
        TrainConfig(max_len=0),
    ],
)
def test_recipe_rejects_nonpositive_training_values(train):
    recipe = Recipe(
        stage="sft", base_model="m", train=train,
        optim=OptimConfig(), io=IOConfig(output_dir="o"),
    )
    with pytest.raises(IRValidationError):
        recipe.validate()


def test_peft_dropout_must_be_less_than_one():
    recipe = _basic_sft_recipe(Path("out"))
    assert recipe.peft is not None
    recipe.peft.dropout = 1.0
    with pytest.raises(IRValidationError, match="< 1"):
        recipe.validate()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_recipe_rejects_non_finite_numbers(value):
    recipe = _basic_sft_recipe(Path("out"))
    recipe.optim.lr = value
    with pytest.raises(IRValidationError, match="finite"):
        recipe.validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scheduler", " ", "optim.scheduler"),
        ("target_template", "", "target_template"),
        ("resume_from", " ", "io.resume_from"),
    ],
)
def test_recipe_rejects_blank_optional_strings(field, value, message):
    recipe = _basic_sft_recipe(Path("out"))
    if field == "scheduler":
        recipe.optim.scheduler = value
    elif field == "resume_from":
        recipe.io.resume_from = value
    else:
        setattr(recipe, field, value)
    with pytest.raises(IRValidationError, match=message):
        recipe.validate()


def test_recipe_type_hints_are_resolvable():
    hints = get_type_hints(PEFTConfig)
    assert hints["target_modules"] == list[str]


def test_recipe_invalid_utf8_is_wrapped(tmp_path):
    path = tmp_path / "bad-utf8.yaml"
    path.write_bytes(b"\xff\n")
    with pytest.raises(IRValidationError, match="bad-utf8.yaml"):
        Recipe.from_yaml(path)
