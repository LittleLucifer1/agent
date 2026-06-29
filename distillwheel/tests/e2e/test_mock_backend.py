"""End-to-end smoke: a mock backend whose 'training' is a `python -c` script.

This exercises the full pipeline (IR validation → adapter resolve →
data write → config write → launcher → log parse → checkpoint
normalize) without needing a GPU or any real ML framework.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from distillwheel.core.adapter import BackendAdapter
from distillwheel.core.checkpoint import CheckpointNormalizer, NormalizedCheckpoint
from distillwheel.core.envspec import EnvSpec
from distillwheel.core.ir.metric import Metric
from distillwheel.core.ir.recipe import IOConfig, OptimConfig, Recipe, TrainConfig
from distillwheel.core.ir.sample import Message, Sample
from distillwheel.core.launcher import Launcher, filter_env
from distillwheel.core.logparser import LogParser
from distillwheel.core.registry import register_adapter, unregister_adapter
from distillwheel.pipeline.orchestrator import run_training


MOCK_TRAIN_SCRIPT = """
import json, os, sys, time
from pathlib import Path

native = Path(os.environ["MOCK_NATIVE_DIR"])
native.mkdir(parents=True, exist_ok=True)

for step in range(1, 4):
    print(f"step={step} loss={1.0/step:.3f} learning_rate=0.0001")
    time.sleep(0.01)

ckdir = native / "checkpoint-3"
ckdir.mkdir(parents=True, exist_ok=True)
(ckdir / "model.safetensors").write_bytes(b"\\x00" * 16)
(ckdir / "config.json").write_text(json.dumps({"_name_or_path": "mock/base"}))
(ckdir / "tokenizer.json").write_text("{}")
print("done")
"""


class _MockParser(LogParser):
    framework = "mock"

    def parse_line(self, line):
        if not line.startswith("step="):
            return None
        # step=1 loss=1.000 learning_rate=0.0001
        parts = dict(p.split("=") for p in line.split())
        return Metric(
            step=int(parts["step"]),
            timestamp=0.0,
            stage=self.stage,
            loss=float(parts["loss"]),
            learning_rate=float(parts["learning_rate"]),
        )


class _MockNormalizer(CheckpointNormalizer):
    framework = "mock"

    def normalize(self, native_dir, output_dir, recipe_yaml_path):
        import shutil

        native_dir = Path(native_dir)
        final_dir = self._ensure_final_dir(output_dir)
        latest = max(native_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        for f in latest.iterdir():
            shutil.copy2(f, final_dir / f.name)
        self._copy_recipe(Path(recipe_yaml_path), final_dir)
        ck = NormalizedCheckpoint(
            final_dir=final_dir,
            is_lora=False,
            step=int(latest.name.split("-")[1]),
            base_model="mock/base",
            framework="mock",
        )
        self._write_metadata(ck)
        return ck


class _MockLauncher(Launcher):
    def __init__(self, env_spec, workdir):
        self.env_spec = env_spec
        self._workdir = Path(workdir)
        self._native = self._workdir / "mock_native"

    def prepare_env(self):
        self._native.mkdir(parents=True, exist_ok=True)

    def command(self):
        return [sys.executable, "-c", MOCK_TRAIN_SCRIPT]

    def env(self):
        return filter_env(extra={"MOCK_NATIVE_DIR": str(self._native)})

    def collect_artifacts(self):
        return self._native


class _MockAdapter(BackendAdapter):
    name = "_mock"
    supported_stages = ("sft",)
    env_spec = EnvSpec(
        venv_path=Path(sys.prefix),
        python_executable=Path(sys.executable),
    )

    def prepare_data(self, stream, recipe, workdir):
        out = Path(workdir) / "data.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for s in stream:
                f.write(json.dumps(s.to_dict()) + "\n")
        return out

    def prepare_config(self, recipe, data_path, workdir):
        cfg = Path(workdir) / "mock.cfg"
        cfg.write_text(f"data={data_path}\n", encoding="utf-8")
        recipe.to_yaml(Path(workdir) / "recipe.yaml")
        return cfg

    def build_launcher(self, config_path, recipe, workdir):
        return _MockLauncher(self.env_spec, workdir)

    def checkpoint_normalizer(self):
        return _MockNormalizer()

    def log_parser(self):
        return _MockParser()


@pytest.fixture
def mock_backend():
    unregister_adapter("_mock")
    register_adapter(_MockAdapter)
    yield
    unregister_adapter("_mock")


def test_end_to_end(mock_backend, tmp_path):
    out_dir = tmp_path / "run"
    recipe = Recipe(
        stage="sft",
        base_model="mock/base",
        train=TrainConfig(),
        optim=OptimConfig(),
        io=IOConfig(output_dir=str(out_dir)),
        backend_hint="_mock",
    )
    samples = [
        Sample(id="s1", task_type="sft",
               messages=[Message(role="user", content="hi"),
                         Message(role="assistant", content="hello")]),
    ]
    root = run_training(recipe, samples, skip_preflight=True)

    assert (root / "final" / "model.safetensors").exists()
    assert (root / "final" / "training_recipe.yaml").exists()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["framework"] == "mock"
    assert metadata["step"] == 3

    metrics_lines = [json.loads(l) for l in (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if l]
    assert len(metrics_lines) == 3
    assert metrics_lines[0]["step"] == 1
    assert metrics_lines[-1]["loss"] == pytest.approx(1 / 3, rel=1e-2)
