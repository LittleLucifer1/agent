"""Keep the checked-in backend smoke recipes aligned with adapter contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from distillwheel.backends.swift.adapter import SwiftAdapter
from distillwheel.backends.swift.recipe_mapper import swift_subcommand_for
from distillwheel.backends.verl.adapter import VerlAdapter
from distillwheel.core.ir.recipe import Recipe
from distillwheel.core.ir.sample import iter_samples_from_jsonl


ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "examples" / "smoke"


@pytest.mark.parametrize(
    ("recipe_file", "data_file", "stage"),
    [
        ("swift_sft.yaml", "swift_sft.jsonl", "sft"),
        ("swift_dpo.yaml", "swift_dpo.jsonl", "dpo"),
    ],
)
def test_swift_smoke_recipe_renders_current_yaml(
    tmp_path, recipe_file, data_file, stage
):
    recipe = Recipe.from_yaml(SMOKE / recipe_file)
    assert recipe.stage == stage
    adapter = SwiftAdapter()
    data_path = adapter.prepare_data(
        iter_samples_from_jsonl(SMOKE / data_file), recipe, tmp_path
    )
    config_path = adapter.prepare_config(recipe, data_path, tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["max_steps"] == 1
    assert config["add_version"] is False
    assert config["split_dataset_ratio"] == 0.0
    assert "__subcommand__" not in config
    assert "sft_type" not in config
    assert "chosen_response" not in data_path.read_text(encoding="utf-8")
    subcommand, _ = swift_subcommand_for(stage)
    command = adapter.build_launcher(config_path, recipe, tmp_path).command()
    assert command[-2:] == [subcommand, str(config_path.resolve())]


@pytest.mark.parametrize(
    ("recipe_file", "stage", "estimator", "critic_enabled"),
    [
        ("verl_grpo.yaml", "grpo", "grpo", "false"),
        ("verl_ppo.yaml", "ppo", "gae", "true"),
        ("verl_rloo.yaml", "rloo", "rloo", "false"),
    ],
)
def test_verl_smoke_recipe_renders_08_contract(
    tmp_path, monkeypatch, recipe_file, stage, estimator, critic_enabled
):
    parquet = pytest.importorskip("pyarrow.parquet")
    monkeypatch.chdir(ROOT)
    recipe = Recipe.from_yaml(SMOKE / recipe_file)
    assert recipe.stage == stage
    adapter = VerlAdapter()
    data_path = adapter.prepare_data(
        iter_samples_from_jsonl(SMOKE / "verl_grpo.jsonl"), recipe, tmp_path
    )
    config_path = adapter.prepare_config(recipe, data_path, tmp_path)
    overrides = dict(
        line.split("=", 1)
        for line in config_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )

    table = parquet.read_table(data_path)
    assert table.num_rows == 2
    assert table.schema.field("prompt").type.value_type.names == ["role", "content"]
    assert overrides["algorithm.adv_estimator"] == estimator
    assert overrides["critic.enable"] == critic_enabled
    assert overrides["trainer.n_gpus_per_node"] == "1"
    assert overrides["trainer.nnodes"] == "1"
    reward_value = overrides["reward.custom_reward_function.path"]
    try:
        reward_value = json.loads(reward_value)
    except json.JSONDecodeError:
        pass
    reward_path = Path(reward_value)
    assert reward_path.is_absolute() and reward_path.samefile(SMOKE / "reward.py")
    assert overrides["reward.custom_reward_function.name"] == "compute_score"
