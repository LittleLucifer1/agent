from pathlib import Path

import pytest

from distillwheel.backends.verl.recipe_mapper import recipe_to_verl_overrides, load_overrides
from distillwheel.core.ir.recipe import (
    IOConfig, OptimConfig, Recipe, RLConfig, TrainConfig,
)


def _grpo_recipe(tmp):
    return Recipe(
        stage="grpo",
        base_model="Qwen/Qwen2-0.5B",
        train=TrainConfig(epochs=1, global_batch=32, micro_batch=1, max_len=4096),
        optim=OptimConfig(lr=1e-6),
        io=IOConfig(output_dir=str(tmp)),
        rl=RLConfig(kl_coef=0.05, clip=0.2, rollout_n=4, rollout_engine="vllm",
                    reward_fn_ref="my_pkg.rewards:math_reward"),
        target_template="qwen",
    )


def test_verl_overrides_grpo(tmp_path):
    r = _grpo_recipe(tmp_path)
    data = tmp_path / "prompts.parquet"
    data.write_text("", encoding="utf-8")
    out = recipe_to_verl_overrides(r, data, tmp_path / "ov.txt")
    overrides = load_overrides(out)
    flat = "\n".join(overrides)
    assert "algorithm=grpo" in flat
    assert "actor_rollout_ref.model.path=Qwen/Qwen2-0.5B" in flat
    assert "actor_rollout_ref.rollout.n=4" in flat
    assert "actor_rollout_ref.actor.kl_loss_coef=0.05" in flat
    assert "reward.custom_reward_function.path=my_pkg.rewards:math_reward" in flat
    assert "actor_rollout_ref.model.chat_template=qwen" in flat


def test_verl_overrides_requires_rl(tmp_path):
    r = Recipe(
        stage="grpo", base_model="m",
        train=TrainConfig(), optim=OptimConfig(),
        io=IOConfig(output_dir=str(tmp_path)),
        # rl missing!
    )
    data = tmp_path / "p.parquet"
    data.write_text("", encoding="utf-8")
    with pytest.raises(Exception):
        recipe_to_verl_overrides(r, data, tmp_path / "ov.txt")


def test_verl_data_writer_streams(tmp_path):
    pa = pytest.importorskip("pyarrow")
    from distillwheel.backends.verl.data_writer import write_verl_parquet
    from distillwheel.core.ir.sample import Sample

    def gen():
        for i in range(2500):
            yield Sample(id=f"q{i}", task_type="rl_prompt", prompt=f"prompt {i}")

    r = _grpo_recipe(tmp_path)
    out = tmp_path / "prompts.parquet"
    write_verl_parquet(gen(), r, out)

    import pyarrow.parquet as pq

    tbl = pq.read_table(out)
    assert tbl.num_rows == 2500
    assert set(tbl.column_names) >= {"id", "prompt", "messages", "meta"}
