import json

import pytest

from distillwheel.core.checkpoint import CheckpointNormalizer, NormalizedCheckpoint
from distillwheel.core.errors import CheckpointError


class _Normalizer(CheckpointNormalizer):
    def normalize(self, native_dir, output_dir, recipe_yaml_path):  # pragma: no cover
        raise NotImplementedError


def test_checkpoint_metadata_roundtrip(tmp_path):
    checkpoint = NormalizedCheckpoint(
        final_dir=tmp_path / "final",
        is_lora=True,
        step=12,
        base_model="base",
        framework="mock",
        extra={"native_dir": "native"},
    )
    checkpoint.final_dir.mkdir()
    _Normalizer()._write_metadata(checkpoint)
    loaded = NormalizedCheckpoint.from_json(checkpoint.final_dir / "metadata.json")
    assert loaded == checkpoint
    assert json.loads((checkpoint.final_dir / "metadata.json").read_text(encoding="utf-8"))["step"] == 12


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("is_lora", "false", "is_lora"),
        ("step", True, "step"),
        ("step", -1, "step"),
        ("framework", "", "framework"),
        ("extra", [], "extra"),
    ],
)
def test_checkpoint_metadata_rejects_invalid_types(tmp_path, field, value, message):
    raw = {
        "final_dir": str(tmp_path / "final"),
        "is_lora": False,
        "step": 1,
        "base_model": "base",
        "framework": "mock",
        "extra": {},
    }
    raw[field] = value
    with pytest.raises(CheckpointError, match=message):
        NormalizedCheckpoint.from_dict(raw)


def test_checkpoint_json_errors_use_domain_exception(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(CheckpointError, match="cannot read checkpoint metadata"):
        NormalizedCheckpoint.from_json(path)


def test_checkpoint_requires_recipe_file(tmp_path):
    with pytest.raises(CheckpointError, match="training recipe"):
        _Normalizer()._copy_recipe(tmp_path / "missing.yaml", tmp_path)


def test_checkpoint_refuses_nonempty_final_directory(tmp_path):
    final = tmp_path / "final"
    final.mkdir()
    (final / "stale-adapter.safetensors").write_bytes(b"stale")
    with pytest.raises(CheckpointError, match="non-empty checkpoint directory"):
        _Normalizer()._ensure_final_dir(tmp_path)
    assert (final / "stale-adapter.safetensors").read_bytes() == b"stale"
