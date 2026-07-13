import json
import os
from pathlib import Path

import pytest
import yaml

from distillwheel.backends.swift.adapter import SwiftAdapter
from distillwheel.backends.swift.checkpoint import SwiftCheckpointNormalizer
from distillwheel.backends.swift.data_writer import sample_to_swift_row, write_swift_jsonl
from distillwheel.backends.swift.launcher import SwiftCLILauncher
from distillwheel.backends.swift.logparser import SwiftLogParser
from distillwheel.backends.swift.recipe_mapper import recipe_to_swift_args
from distillwheel.core.envspec import EnvSpec
from distillwheel.core.errors import CheckpointError, IRValidationError
from distillwheel.core.ir.recipe import (
    IOConfig,
    OptimConfig,
    PEFTConfig,
    ParallelConfig,
    Recipe,
    TrainConfig,
)
from distillwheel.core.ir.sample import Message, Sample


def _recipe(tmp_path: Path, *, stage="sft", **kwargs) -> Recipe:
    values = {
        "stage": stage,
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "train": TrainConfig(),
        "optim": OptimConfig(),
        "io": IOConfig(output_dir=str(tmp_path)),
    }
    values.update(kwargs)
    return Recipe(**values)


def _sft_sample(i):
    return Sample(
        id=f"s{i}",
        task_type="sft",
        messages=[
            Message(role="user", content="q"),
            Message(role="assistant", content="a"),
        ],
    )


def test_swift_sft_row_preserves_tools_images_and_message_fields():
    sample = _sft_sample(1)
    sample.messages[1].tool_calls = [{"id": "call-1", "type": "function"}]
    sample.tools = [{"type": "function", "function": {"name": "lookup"}}]
    sample.images = ["image.jpg"]

    row = sample_to_swift_row(sample, "sft")

    assert row["messages"][0]["role"] == "user"
    assert row["messages"][1]["tool_calls"][0]["id"] == "call-1"
    assert row["tools"] == sample.tools
    assert row["images"] == ["image.jpg"]


def test_swift_dpo_row_appends_chosen_and_has_only_rejected_response():
    sample = Sample(
        id="d1",
        task_type="preference",
        prompt="q",
        chosen="good",
        rejected="bad",
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        images=["image.jpg"],
    )

    row = sample_to_swift_row(sample, "dpo")

    assert [message["role"] for message in row["messages"]] == ["user", "assistant"]
    assert row["messages"][-1]["content"] == "good"
    assert row["rejected_response"] == "bad"
    assert "chosen_response" not in row
    assert row["tools"] == sample.tools
    assert row["images"] == sample.images


def test_swift_dpo_row_preserves_structured_chosen_messages():
    sample = Sample(
        id="d2",
        task_type="preference",
        prompt=[Message(role="system", content="be useful"), Message(role="user", content="q")],
        chosen=[
            Message(role="assistant", content="calling", tool_calls=[{"id": "c1"}]),
            Message(role="tool", content="result", tool_call_id="c1"),
            Message(role="assistant", content="good"),
        ],
        rejected=[Message(role="assistant", content="bad")],
    )

    row = sample_to_swift_row(sample, "dpo")

    assert [message["role"] for message in row["messages"]] == [
        "system", "user", "assistant", "tool", "assistant"
    ]
    assert row["messages"][2]["tool_calls"] == [{"id": "c1"}]
    assert row["rejected_response"] == "bad"


def test_swift_kto_row_and_label_validation():
    sample = Sample(id="k1", task_type="kto", prompt="q", completion="a", label=1)
    row = sample_to_swift_row(sample, "kto")
    assert row["label"] is True
    assert row["messages"][-1]["role"] == "assistant"

    invalid = Sample(id="k2", task_type="kto", prompt="q", completion="a", label=2)
    with pytest.raises(IRValidationError, match="label"):
        sample_to_swift_row(invalid, "kto")


def test_swift_write_jsonl(tmp_path):
    out = tmp_path / "data.jsonl"
    write_swift_jsonl((_sft_sample(i) for i in range(3)), _recipe(tmp_path), out)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert rows[0]["id"] == "s0"


def test_swift_recipe_mapper_emits_only_ms_swift_4_keys(tmp_path):
    recipe = _recipe(
        tmp_path,
        train=TrainConfig(epochs=2, micro_batch=2, grad_accum=4, max_len=2048),
        optim=OptimConfig(lr=1e-5),
        io=IOConfig(output_dir="ignored-by-native-workdir", save_steps=1),
        peft=PEFTConfig(type="lora", r=8, alpha=16, target_modules=["q_proj", "v_proj"]),
        target_template="qwen",
    )
    data = tmp_path / "data.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    cfg_path = tmp_path / "run" / "swift.yaml"

    written = recipe_to_swift_args(recipe, data, cfg_path)
    parsed = yaml.safe_load(written.read_text(encoding="utf-8"))

    emitted_key_allowlist = {
        "model", "dataset", "output_dir", "num_train_epochs",
        "per_device_train_batch_size", "gradient_accumulation_steps",
        "learning_rate", "lr_scheduler_type", "warmup_ratio", "weight_decay",
        "max_length", "seed", "save_strategy", "save_steps", "logging_steps",
        "split_dataset_ratio", "add_version", "torch_dtype", "tuner_type",
        "lora_rank", "lora_alpha", "lora_dropout", "target_modules", "template",
    }
    assert set(parsed) <= emitted_key_allowlist
    assert parsed["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert parsed["tuner_type"] == "lora"
    assert parsed["target_modules"] == ["q_proj", "v_proj"]
    assert parsed["torch_dtype"] == "bfloat16"
    assert parsed["save_strategy"] == "steps"
    assert parsed["save_steps"] == 1
    assert parsed["split_dataset_ratio"] == 0.0
    assert parsed["add_version"] is False
    assert parsed["dataset"] == [str(data.resolve())]
    assert parsed["output_dir"] == str((cfg_path.parent / "swift_native").resolve())
    assert written.is_absolute()
    for obsolete in ("sft_type", "lora_target_modules", "bf16", "fp16", "__subcommand__"):
        assert obsolete not in parsed


def test_swift_recipe_mapper_dpo_rlhf_and_fp16(tmp_path):
    recipe = _recipe(tmp_path, stage="dpo", precision="fp16")
    data = tmp_path / "data.jsonl"
    data.write_text("{}\n", encoding="utf-8")

    parsed = yaml.safe_load(
        recipe_to_swift_args(recipe, data, tmp_path / "swift.yaml").read_text(encoding="utf-8")
    )

    assert parsed["rlhf_type"] == "dpo"
    assert parsed["tuner_type"] == "full"
    assert parsed["torch_dtype"] == "float16"
    assert "__subcommand__" not in parsed


def test_swift_recipe_mapper_qlora_uses_bnb_lora(tmp_path):
    recipe = _recipe(
        tmp_path,
        peft=PEFTConfig(type="qlora", r=4, alpha=8, target_modules=["all-linear"]),
    )
    data = tmp_path / "data.jsonl"
    data.write_text("{}\n", encoding="utf-8")

    parsed = yaml.safe_load(
        recipe_to_swift_args(recipe, data, tmp_path / "swift.yaml").read_text(encoding="utf-8")
    )

    assert parsed["tuner_type"] == "lora"
    assert parsed["quant_method"] == "bnb"
    assert parsed["quant_bits"] == 4
    assert parsed["target_modules"] == ["all-linear"]


def test_swift_recipe_mapper_rejects_fp8_and_tp_pp(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    with pytest.raises(IRValidationError, match="precision='fp8'"):
        recipe_to_swift_args(
            _recipe(tmp_path, precision="fp8"), data, tmp_path / "fp8.yaml"
        )
    with pytest.raises(IRValidationError, match="Megatron-SWIFT"):
        recipe_to_swift_args(
            _recipe(tmp_path, parallel=ParallelConfig(tp=2)),
            data,
            tmp_path / "tp.yaml",
        )
    with pytest.raises(IRValidationError, match="Megatron-SWIFT"):
        recipe_to_swift_args(
            _recipe(tmp_path, parallel=ParallelConfig(pp=2)),
            data,
            tmp_path / "pp.yaml",
        )


def test_swift_recipe_mapper_resume_deepspeed_and_controlled_overrides(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    resume = tmp_path / "old" / "checkpoint-7"
    recipe = _recipe(
        tmp_path,
        io=IOConfig(output_dir=str(tmp_path), resume_from=str(resume)),
        parallel=ParallelConfig(zero_stage=3),
        meta={"swift": {"overrides": {"split_dataset_ratio": 0.2, "report_to": "none"}}},
    )

    parsed = yaml.safe_load(
        recipe_to_swift_args(recipe, data, tmp_path / "swift.yaml").read_text(encoding="utf-8")
    )

    assert parsed["resume_from_checkpoint"] == str(resume.resolve())
    assert parsed["deepspeed"] == "zero3"
    assert parsed["split_dataset_ratio"] == 0.2
    assert parsed["report_to"] == "none"

    recipe.meta["swift"]["overrides"] = {"dataset": ["other.jsonl"]}
    with pytest.raises(IRValidationError, match="DistillWheel-owned"):
        recipe_to_swift_args(recipe, data, tmp_path / "forbidden.yaml")


def _launcher_spec(tmp_path: Path, *, with_console_script: bool) -> tuple[EnvSpec, Path | None]:
    bin_dir = tmp_path / "venv" / ("Scripts" if os.name == "nt" else "bin")
    bin_dir.mkdir(parents=True)
    python = bin_dir / ("python.exe" if os.name == "nt" else "python")
    python.write_bytes(b"")
    python.chmod(0o755)
    swift = bin_dir / ("swift.exe" if os.name == "nt" else "swift")
    if with_console_script:
        swift.write_bytes(b"")
        swift.chmod(0o755)
    spec = EnvSpec(venv_path=tmp_path / "venv", python_executable=python)
    return spec, swift if with_console_script else None


def test_swift_launcher_uses_positional_yaml_and_recipe_stage(tmp_path):
    spec, swift = _launcher_spec(tmp_path, with_console_script=True)
    config = tmp_path / "cfg" / "swift.yaml"
    config.parent.mkdir()
    config.write_text("model: m\n", encoding="utf-8")
    launcher = SwiftCLILauncher(spec, config, _recipe(tmp_path), tmp_path / "work")

    command = launcher.command()

    assert command == [str(swift.resolve()), "sft", str(config.resolve())]
    assert "--config" not in command


def test_swift_launcher_fallback_is_4x_cli_module(tmp_path):
    spec, _ = _launcher_spec(tmp_path, with_console_script=False)
    config = tmp_path / "swift.yaml"
    config.write_text("model: m\n", encoding="utf-8")
    launcher = SwiftCLILauncher(
        spec, config, _recipe(tmp_path, stage="dpo"), tmp_path / "work"
    )

    assert launcher.command() == [
        str(spec.python_executable),
        "-m",
        "swift.cli.main",
        "rlhf",
        str(config.resolve()),
    ]


def test_swift_launcher_sets_multigpu_env_from_dp(tmp_path, monkeypatch):
    spec, _ = _launcher_spec(tmp_path, with_console_script=False)
    config = tmp_path / "swift.yaml"
    config.write_text("model: m\n", encoding="utf-8")
    monkeypatch.setenv("NPROC_PER_NODE", "99")
    monkeypatch.setenv("NODE_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "99")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    launcher = SwiftCLILauncher(
        spec,
        config,
        _recipe(tmp_path, parallel=ParallelConfig(dp=4)),
        tmp_path / "work",
    )

    child_env = launcher.env()

    assert child_env["NPROC_PER_NODE"] == "4"
    assert "NNODES" not in child_env
    assert "NODE_RANK" not in child_env
    assert "WORLD_SIZE" not in child_env
    assert child_env["MASTER_ADDR"] == "10.0.0.1"


def test_swift_launcher_uses_plain_process_for_single_gpu(tmp_path, monkeypatch):
    spec, _ = _launcher_spec(tmp_path, with_console_script=False)
    config = tmp_path / "swift.yaml"
    config.write_text("model: m\n", encoding="utf-8")
    monkeypatch.setenv("NPROC_PER_NODE", "99")
    launcher = SwiftCLILauncher(
        spec,
        config,
        _recipe(tmp_path, parallel=ParallelConfig(dp=1)),
        tmp_path / "work",
    )

    assert "NPROC_PER_NODE" not in launcher.env()


def test_swift_launcher_rejects_unmodelled_multinode_env(tmp_path, monkeypatch):
    spec, _ = _launcher_spec(tmp_path, with_console_script=False)
    config = tmp_path / "swift.yaml"
    config.write_text("model: m\n", encoding="utf-8")
    monkeypatch.setenv("NNODES", "2")
    launcher = SwiftCLILauncher(
        spec,
        config,
        _recipe(tmp_path, parallel=ParallelConfig(dp=4)),
        tmp_path / "work",
    )

    with pytest.raises(IRValidationError, match="ResourceConfig"):
        launcher.env()


def test_swift_adapter_pins_4_4_and_resolves_environment_paths():
    assert "ms-swift>=4.4,<4.5" in SwiftAdapter.env_spec.required_packages
    assert SwiftAdapter.env_spec.venv_path.is_absolute()
    assert SwiftAdapter.env_spec.python_executable.is_absolute()
    assert SwiftAdapter.env_spec.health_check_cmd
    assert "ms-swift" in SwiftAdapter.env_spec.health_check_cmd[-1]
    assert "trl" in SwiftAdapter.env_spec.health_check_cmd[-1]


def test_swift_checkpoint_normalizes_sharded_full_model(tmp_path):
    native = tmp_path / "native"
    checkpoint = native / "checkpoint-12"
    checkpoint.mkdir(parents=True)
    shards = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    for index, shard in enumerate(shards):
        (checkpoint / shard).write_text(f"shard-{index}", encoding="utf-8")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a.weight": shards[0], "b.weight": shards[1]}}),
        encoding="utf-8",
    )
    (checkpoint / "config.json").write_text(
        json.dumps({"_name_or_path": "base/model"}), encoding="utf-8"
    )
    asset_names = (
        "added_tokens.json",
        "vocab.txt",
        "spiece.model",
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
    )
    for name in asset_names:
        (checkpoint / name).write_text(f"asset:{name}", encoding="utf-8")
    # ms-swift may keep args.json at the output root rather than in every
    # checkpoint; ancestor discovery should preserve it as provenance.
    (native / "args.json").write_text("{}", encoding="utf-8")
    recipe_yaml = tmp_path / "recipe.yaml"
    recipe_yaml.write_text("stage: sft\n", encoding="utf-8")

    normalized = SwiftCheckpointNormalizer().normalize(
        native, tmp_path / "output", recipe_yaml
    )

    assert normalized.step == 12
    assert normalized.is_lora is False
    assert normalized.base_model == "base/model"
    assert (normalized.final_dir / "model.safetensors.index.json").is_file()
    assert all((normalized.final_dir / shard).is_file() for shard in shards)
    assert all((normalized.final_dir / name).is_file() for name in asset_names)
    assert (normalized.final_dir / "args.json").is_file()


def test_swift_checkpoint_rejects_missing_index_shard(tmp_path):
    checkpoint = tmp_path / "native" / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a.weight": "missing.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(CheckpointError, match="missing shard"):
        SwiftCheckpointNormalizer().normalize(
            tmp_path / "native", tmp_path / "output", tmp_path / "recipe.yaml"
        )


def test_swift_checkpoint_finds_latest_nested_version_checkpoint(tmp_path):
    native = tmp_path / "native"
    for version, step in (("v0-old", 3), ("v1-new", 9)):
        checkpoint = native / version / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_model.safetensors").write_text(
            f"step-{step}", encoding="utf-8"
        )
        (checkpoint / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "base/model"}), encoding="utf-8"
        )
    recipe_yaml = tmp_path / "recipe.yaml"
    recipe_yaml.write_text("stage: sft\n", encoding="utf-8")

    normalized = SwiftCheckpointNormalizer().normalize(
        native, tmp_path / "output", recipe_yaml
    )

    assert normalized.step == 9
    assert normalized.is_lora is True
    assert (normalized.final_dir / "adapter_model.safetensors").read_text(
        encoding="utf-8"
    ) == "step-9"
    assert (tmp_path / "output" / "checkpoints" / "v1-new" / "checkpoint-9").is_dir()


def test_swift_logparser_handles_real_4x_progress_dict():
    parser = SwiftLogParser()
    parser.stage = "dpo"
    metric = parser.parse_line(
        "[INFO:swift] {'loss': 1.41926861, 'grad_norm': 8.33, "
        "'learning_rate': 9e-06, 'rewards/chosen': -0.06, "
        "'rewards/rejected': -0.1, 'epoch': 0.03, "
        "'global_step/max_steps': '1/840', 'percentage': '0.12%'}"
    )

    assert metric is not None
    assert metric.step == 1
    assert metric.loss == pytest.approx(1.41926861)
    assert metric.learning_rate == pytest.approx(9e-06)
    assert metric.grad_norm == pytest.approx(8.33)
    assert metric.extra["rewards/chosen"] == -0.06


def test_swift_logparser_handles_namespaced_4x_keys():
    parser = SwiftLogParser()
    metric = parser.parse_line(
        "{'train/loss': 0.5, 'train/learning_rate': 0.0001, "
        "'train/grad_norm': 1.25, 'train/epoch': 0.5, "
        "'rewards/mean': 0.75, 'global_step': 4}"
    )

    assert metric is not None
    assert metric.step == 4
    assert metric.loss == 0.5
    assert metric.learning_rate == 0.0001
    assert metric.grad_norm == 1.25
    assert metric.epoch == 0.5
    assert metric.reward_mean == 0.75
