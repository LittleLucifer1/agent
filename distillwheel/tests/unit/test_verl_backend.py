import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from distillwheel.backends.verl.adapter import VerlAdapter
from distillwheel.backends.verl.checkpoint import VerlCheckpointNormalizer
from distillwheel.backends.verl.launcher import VerlRayLauncher
from distillwheel.backends.verl.logparser import VerlLogParser
from distillwheel.backends.verl.recipe_mapper import (
    load_overrides,
    recipe_to_verl_overrides,
    verl_algorithm_for,
)
from distillwheel.core.envspec import EnvSpec
from distillwheel.core.errors import CheckpointError, IRValidationError
from distillwheel.core.ir.recipe import (
    IOConfig,
    OptimConfig,
    PEFTConfig,
    ParallelConfig,
    Recipe,
    RLConfig,
    TrainConfig,
)


def _recipe(tmp_path: Path, stage: str = "grpo") -> Recipe:
    return Recipe(
        stage=stage,
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        train=TrainConfig(epochs=1, global_batch=8, micro_batch=2, max_len=1024),
        optim=OptimConfig(
            lr=1e-6, scheduler="cosine", warmup_ratio=0.03, weight_decay=0.01
        ),
        io=IOConfig(output_dir=str(tmp_path), save_steps=5, logging_steps=1),
        rl=RLConfig(
            kl_coef=0.05,
            clip=0.2,
            rollout_n=4,
            rollout_engine="vllm",
            reward_fn_ref="my_pkg.rewards:math_reward",
        ),
        target_template="qwen",  # A name must not be sent as Jinja.
    )


def _override_map(path: Path) -> dict:
    return dict(item.split("=", 1) for item in load_overrides(path))


def _decoded(value: str):
    if value.startswith(('"', "[", "{")):
        return json.loads(value)
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "null":
        return None
    return value


def _map(tmp_path: Path, recipe: Recipe) -> dict:
    data = tmp_path / "prompts.parquet"
    data.write_bytes(b"parquet-placeholder")
    path = recipe_to_verl_overrides(recipe, data, tmp_path / "verl_overrides.txt")
    return _override_map(path)


def test_verl_supports_only_public_08_algorithms():
    assert VerlAdapter.supported_stages == ("grpo", "ppo", "rloo")
    assert verl_algorithm_for("grpo") == "grpo"
    assert verl_algorithm_for("ppo") == "gae"
    assert verl_algorithm_for("rloo") == "rloo"
    with pytest.raises(IRValidationError, match="supports only"):
        verl_algorithm_for("opd")


def test_grpo_overrides_match_verl_08_schema(tmp_path):
    overrides = _map(tmp_path, _recipe(tmp_path))

    assert overrides["algorithm.adv_estimator"] == "grpo"
    assert overrides["actor_rollout_ref.model.path"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert overrides["actor_rollout_ref.rollout.name"] == "vllm"
    assert overrides["actor_rollout_ref.rollout.n"] == "4"
    assert overrides["actor_rollout_ref.rollout.dtype"] == "bfloat16"
    assert overrides["actor_rollout_ref.actor.fsdp_config.dtype"] == "bfloat16"
    assert overrides["actor_rollout_ref.ref.fsdp_config.dtype"] == "bfloat16"
    assert overrides["actor_rollout_ref.actor.ppo_mini_batch_size"] == "8"
    assert overrides["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == "2"
    assert overrides["actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"] == "2"
    assert overrides["actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu"] == "2"
    assert overrides["actor_rollout_ref.actor.use_kl_loss"] == "true"
    assert overrides["actor_rollout_ref.actor.kl_loss_coef"] == "0.05"
    assert overrides["algorithm.use_kl_in_reward"] == "false"
    assert overrides["critic.enable"] == "false"
    assert overrides["trainer.val_before_train"] == "false"
    assert overrides["trainer.test_freq"] == "-1"
    assert overrides["trainer.n_gpus_per_node"] == "1"
    assert overrides["trainer.nnodes"] == "1"
    assert _decoded(overrides["data.train_files"]) == str((tmp_path / "prompts.parquet").resolve())
    assert overrides["data.train_files"] == overrides["data.val_files"]

    # These were emitted by the original mapper but are not VERL 0.8 keys.
    forbidden = {
        "algorithm",
        "trainer.seed",
        "trainer.logger.console.log_freq",
        "actor_rollout_ref.model.torch_dtype",
        "actor_rollout_ref.model.chat_template",
        "trainer.resume_from",
    }
    assert forbidden.isdisjoint(overrides)


def test_ppo_adds_critic_and_reward_kl_config(tmp_path):
    recipe = _recipe(tmp_path, "ppo")
    recipe.rl = replace(recipe.rl, rollout_n=1)
    overrides = _map(tmp_path, recipe)

    assert overrides["algorithm.adv_estimator"] == "gae"
    assert overrides["critic.enable"] == "true"
    assert overrides["critic.model.path"] == recipe.base_model
    assert overrides["critic.ppo_mini_batch_size"] == "8"
    assert overrides["critic.ppo_micro_batch_size_per_gpu"] == "2"
    assert overrides["critic.optim.lr"] == "1e-06"
    assert overrides["critic.fsdp.dtype"] == "bfloat16"
    assert overrides["actor_rollout_ref.actor.use_kl_loss"] == "false"
    assert overrides["algorithm.use_kl_in_reward"] == "true"
    assert overrides["algorithm.kl_ctrl.kl_coef"] == "0.05"


def test_fp16_precision_reaches_rollout_actor_ref_and_critic(tmp_path):
    recipe = _recipe(tmp_path, "ppo")
    recipe.precision = "fp16"
    recipe.rl = replace(recipe.rl, rollout_n=1)
    overrides = _map(tmp_path, recipe)

    assert overrides["actor_rollout_ref.rollout.dtype"] == "float16"
    assert overrides["actor_rollout_ref.actor.fsdp_config.dtype"] == "float16"
    assert overrides["actor_rollout_ref.ref.fsdp_config.dtype"] == "float16"
    assert overrides["critic.fsdp.dtype"] == "float16"


def test_rloo_requires_multiple_rollouts(tmp_path):
    recipe = _recipe(tmp_path, "rloo")
    recipe.rl = replace(recipe.rl, rollout_n=1)
    with pytest.raises(IRValidationError, match="rollout_n >= 2"):
        _map(tmp_path, recipe)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda r: replace(r, precision="fp8"), "fp8"),
        (lambda r: replace(r, parallel=ParallelConfig(zero_stage=2)), "zero_stage"),
        (lambda r: replace(r, parallel=ParallelConfig(pp=2)), "pp=1"),
        (lambda r: replace(r, peft=PEFTConfig(type="qlora")), "QLoRA"),
        (lambda r: replace(r, peft=PEFTConfig(dropout=0.1)), "dropout must be 0"),
    ],
)
def test_unmappable_recipe_settings_fail_before_launch(tmp_path, change, message):
    with pytest.raises(IRValidationError, match=message):
        _map(tmp_path, change(_recipe(tmp_path)))


def test_meta_override_is_last_unique_and_controlled(tmp_path):
    recipe = _recipe(tmp_path)
    recipe.meta = {
        "verl": {
            "overrides": {
                "actor_rollout_ref.rollout.gpu_memory_utilization": 0.55,
                "trainer.project_name": "distillwheel-smoke",
            }
        }
    }
    data = tmp_path / "prompts.parquet"
    data.write_bytes(b"x")
    path = recipe_to_verl_overrides(recipe, data, tmp_path / "overrides.txt")
    lines = load_overrides(path)
    overrides = _override_map(path)
    assert overrides["actor_rollout_ref.rollout.gpu_memory_utilization"] == "0.55"
    assert len(lines) == len(overrides)  # no duplicate Hydra keys

    recipe.meta = {"verl": {"overrides": {"algorithm": "grpo"}}}
    with pytest.raises(IRValidationError, match="forbidden or malformed"):
        recipe_to_verl_overrides(recipe, data, tmp_path / "bad.txt")

    recipe.meta = {"verl": {"overrides": {"data.train_files": "elsewhere"}}}
    with pytest.raises(IRValidationError, match="managed by DistillWheel"):
        recipe_to_verl_overrides(recipe, data, tmp_path / "bad-protected.txt")


def test_custom_chat_template_must_be_jinja(tmp_path):
    recipe = _recipe(tmp_path)
    recipe.meta = {"verl": {"custom_chat_template": "qwen"}}
    with pytest.raises(IRValidationError, match="Jinja"):
        _map(tmp_path, recipe)

    template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    recipe.meta = {"verl": {"custom_chat_template": template}}
    overrides = _map(tmp_path, recipe)
    assert _decoded(overrides["actor_rollout_ref.model.custom_chat_template"]) == template


def test_dotted_reward_reference_generates_verl_file_shim(tmp_path):
    overrides = _map(tmp_path, _recipe(tmp_path))
    shim = Path(_decoded(overrides["reward.custom_reward_function.path"]))
    assert shim.is_absolute() and shim.is_file()
    assert overrides["reward.custom_reward_function.name"] == "math_reward"
    source = shim.read_text(encoding="utf-8")
    assert "my_pkg.rewards" in source
    assert "math_reward" in source


def test_reward_file_is_resolved_to_absolute_path_and_validated(tmp_path, monkeypatch):
    reward = (tmp_path / "reward.py").resolve()
    reward.write_text("def score(*args, **kwargs): return 1\n", encoding="utf-8")
    recipe = _recipe(tmp_path)
    recipe.rl = replace(recipe.rl, reward_fn_ref=f"{reward}:score")
    overrides = _map(tmp_path, recipe)
    assert _decoded(overrides["reward.custom_reward_function.path"]) == str(reward)
    assert overrides["reward.custom_reward_function.name"] == "score"

    monkeypatch.chdir(tmp_path)
    recipe.rl = replace(recipe.rl, reward_fn_ref="reward.py:score")
    overrides = _map(tmp_path, recipe)
    assert _decoded(overrides["reward.custom_reward_function.path"]) == str(reward)

    recipe.rl = replace(recipe.rl, reward_fn_ref="missing_reward.py:score")
    with pytest.raises(IRValidationError, match="does not exist"):
        _map(tmp_path, recipe)


def test_verl_parquet_uses_nested_native_columns(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from distillwheel.backends.verl.data_writer import write_verl_parquet
    from distillwheel.core.ir.sample import Message, Sample

    def samples():
        for index in range(1025):
            yield Sample(
                id=f"q{index}",
                task_type="rl_prompt",
                prompt=[Message(role="system", content="Be concise"), Message(role="user", content=f"{index}+1?")],
                meta={
                    "data_source": "unit/math",
                    "ability": "arithmetic",
                    "reward_model": {"style": "rule", "ground_truth": str(index + 1)},
                    "extra_info": {"index": index, "split": "train", "question": f"{index}+1?"},
                    "difficulty": "easy",
                },
            )

    output = write_verl_parquet(samples(), _recipe(tmp_path), tmp_path / "prompts.parquet")
    table = pq.read_table(output)
    assert table.num_rows == 1025
    assert table.column_names == [
        "data_source", "prompt", "ability", "reward_model", "extra_info"
    ]
    prompt_type = table.schema.field("prompt").type
    assert pa.types.is_list(prompt_type)
    assert pa.types.is_struct(prompt_type.value_type)
    assert prompt_type.value_type.names == ["role", "content"]
    assert pa.types.is_struct(table.schema.field("reward_model").type)
    assert pa.types.is_struct(table.schema.field("extra_info").type)
    assert table["prompt"][0].as_py()[1] == {"role": "user", "content": "0+1?"}
    assert table["reward_model"][0].as_py()["ground_truth"] == "1"
    extra = table["extra_info"][0].as_py()
    assert extra["sample_id"] == "q0"
    assert {entry["key"]: entry["value"] for entry in extra["metadata"]}["difficulty"] == "easy"


def test_verl_parquet_rejects_empty_and_non_rl_data(tmp_path):
    pytest.importorskip("pyarrow")
    from distillwheel.backends.verl.data_writer import write_verl_parquet
    from distillwheel.core.ir.sample import Message, Sample

    with pytest.raises(IRValidationError, match="empty"):
        write_verl_parquet([], _recipe(tmp_path), tmp_path / "empty.parquet")
    bad = Sample(id="s1", task_type="sft", messages=[Message("user", "hi")])
    with pytest.raises(IRValidationError, match="only task_type='rl_prompt'"):
        write_verl_parquet([bad], _recipe(tmp_path), tmp_path / "bad.parquet")


@pytest.mark.parametrize("field", ["images", "tools"])
def test_verl_parquet_refuses_unmapped_multimodal_or_tool_fields(tmp_path, field):
    pytest.importorskip("pyarrow")
    from distillwheel.backends.verl.data_writer import write_verl_parquet
    from distillwheel.core.ir.sample import Sample

    sample = Sample(id="rl-unsupported", task_type="rl_prompt", prompt="hello")
    setattr(sample, field, ["image.png"] if field == "images" else [{"name": "tool"}])

    with pytest.raises(IRValidationError, match="refusing to drop"):
        write_verl_parquet([sample], _recipe(tmp_path), tmp_path / "unsupported.parquet")


def test_launcher_always_uses_absolute_backend_python_even_with_ray_address(tmp_path, monkeypatch):
    python = tmp_path / ("python.exe" if __import__("os").name == "nt" else "python")
    python.write_text("", encoding="utf-8")
    python.chmod(0o755)
    env_spec = EnvSpec(tmp_path / "env", python, required_packages=[])
    overrides = tmp_path / "overrides.txt"
    overrides.write_text("algorithm.adv_estimator=grpo\n", encoding="utf-8")
    monkeypatch.setenv("RAY_ADDRESS", "ray://cluster:10001")

    launcher = VerlRayLauncher(env_spec, overrides, _recipe(tmp_path), tmp_path)
    command = launcher.command()
    assert command[:3] == [str(python.resolve()), "-m", "verl.trainer.main_ppo"]
    assert "ray" not in command[:1]
    assert launcher.env()["RAY_ADDRESS"] == "ray://cluster:10001"


def test_checkpoint_merger_uses_verl_08_cli(tmp_path, monkeypatch):
    python = tmp_path / ("python.exe" if __import__("os").name == "nt" else "python")
    python.write_text("", encoding="utf-8")
    python.chmod(0o755)
    actor = tmp_path / "global_step_9" / "actor"
    final = tmp_path / "final"
    actor.mkdir(parents=True)
    final.mkdir()
    env_spec = EnvSpec(tmp_path / "env", python, required_packages=[])
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="merged", stderr="")

    monkeypatch.setattr("distillwheel.backends.verl.checkpoint.subprocess.run", fake_run)
    VerlCheckpointNormalizer(env_spec)._merge_with_verl_script(actor, final)
    assert captured["command"] == [
        str(python.resolve()), "-m", "verl.model_merger", "merge",
        "--backend", "fsdp", "--local_dir", str(actor.resolve()),
        "--target_dir", str(final.resolve()),
    ]


def test_checkpoint_merger_error_contains_stdout_and_stderr(tmp_path, monkeypatch):
    python = tmp_path / ("python.exe" if __import__("os").name == "nt" else "python")
    python.write_text("", encoding="utf-8")
    python.chmod(0o755)
    actor, final = tmp_path / "actor", tmp_path / "final"
    actor.mkdir(); final.mkdir()
    env_spec = EnvSpec(tmp_path / "env", python, required_packages=[])
    monkeypatch.setattr(
        "distillwheel.backends.verl.checkpoint.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=2, stdout="OUT", stderr="ERR"),
    )
    with pytest.raises(CheckpointError) as error:
        VerlCheckpointNormalizer(env_spec)._merge_with_verl_script(actor, final)
    assert "OUT" in str(error.value) and "ERR" in str(error.value)


def test_premerged_actor_huggingface_checkpoint_skips_merger(tmp_path):
    source = tmp_path / "native" / "global_step_3" / "actor" / "huggingface"
    source.mkdir(parents=True)
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "config.json").write_text('{"_name_or_path":"checkpoint/model"}', encoding="utf-8")
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")
    (source / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (source / "processor_config.json").write_text("{}", encoding="utf-8")
    (source / "video_preprocessor_config.json").write_text("{}", encoding="utf-8")
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("base_model: recipe/model\n", encoding="utf-8")

    checkpoint = VerlCheckpointNormalizer().normalize(
        tmp_path / "native", tmp_path / "output", recipe
    )
    assert checkpoint.step == 3
    assert checkpoint.base_model == "recipe/model"
    assert (checkpoint.final_dir / "model.safetensors").is_file()
    assert (checkpoint.final_dir / "tokenizer.json").is_file()
    assert (checkpoint.final_dir / "preprocessor_config.json").is_file()
    assert (checkpoint.final_dir / "processor_config.json").is_file()
    assert (checkpoint.final_dir / "video_preprocessor_config.json").is_file()
    assert not (checkpoint.final_dir / "BASE_MODEL_REQUIRED.txt").exists()


def test_lora_checkpoint_records_required_base_model(tmp_path):
    source = tmp_path / "native" / "global_step_1" / "actor" / "huggingface"
    source.mkdir(parents=True)
    (source / "adapter_model.safetensors").write_bytes(b"adapter")
    (source / "adapter_config.json").write_text(
        '{"base_model_name_or_path":"Qwen/Qwen2.5-0.5B-Instruct"}',
        encoding="utf-8",
    )
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "base_model: Qwen/Qwen2.5-0.5B-Instruct\n",
        encoding="utf-8",
    )

    checkpoint = VerlCheckpointNormalizer().normalize(
        tmp_path / "native", tmp_path / "output", recipe
    )

    assert checkpoint.is_lora is True
    requirement = (checkpoint.final_dir / "BASE_MODEL_REQUIRED.txt").read_text(
        encoding="utf-8"
    )
    assert "Base model required" in requirement
    assert "Qwen/Qwen2.5-0.5B-Instruct" in requirement


def test_official_verl_08_ray_console_sample_is_parsed():
    parser = VerlLogParser()
    parser.stage = "grpo"
    metric = parser.parse_line(
        "\x1b[36m(TaskRunner pid=2468)\x1b[0m step:12 - actor/pg_loss:-5.0e-3 "
        "- actor/ppo_kl:1.2e-4 - critic/score/mean:0.8 - critic/score/std:0.1 "
        "- critic/vf_loss:0.03 - actor/entropy_loss:0.2 - actor/lr(AdamW):1e-6"
    )
    assert metric is not None
    assert metric.step == 12
    assert metric.loss == metric.policy_loss == -0.005
    assert metric.kl == pytest.approx(1.2e-4)
    assert metric.reward_mean == 0.8
    assert metric.reward_std == 0.1
    assert metric.value_loss == 0.03
    assert metric.entropy == 0.2
    assert metric.learning_rate == 1e-6
