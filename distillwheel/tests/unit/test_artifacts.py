import json

import pytest

from distillwheel.core.errors import OutputDirectoryError
from distillwheel.pipeline.artifacts import build_output_layout


def test_nonempty_output_is_rejected_by_default(tmp_path):
    root = tmp_path / "run"
    with build_output_layout(root):
        pass
    with pytest.raises(OutputDirectoryError, match="not empty"):
        build_output_layout(root)


def test_overwrite_only_cleans_marked_managed_paths(tmp_path):
    root = tmp_path / "run"
    with build_output_layout(root) as first:
        first.append_raw_log("old-run")
        (first.final_dir / "old-model.bin").write_bytes(b"old")
    note = root / "notes.txt"
    note.write_text("keep", encoding="utf-8")

    with build_output_layout(root, overwrite=True) as second:
        assert second.tail_raw_log() == ""
        assert not (second.final_dir / "old-model.bin").exists()
        assert note.read_text(encoding="utf-8") == "keep"


def test_overwrite_refuses_unmarked_directory(tmp_path):
    root = tmp_path / "not-owned"
    root.mkdir()
    (root / "workdir").mkdir()
    with pytest.raises(OutputDirectoryError, match="unmarked"):
        build_output_layout(root, overwrite=True)


def test_overwrite_refuses_forged_or_corrupt_marker(tmp_path):
    root = tmp_path / "not-owned"
    root.mkdir()
    marker = root / ".distillwheel-run.json"
    marker.write_text("not json", encoding="utf-8")
    with pytest.raises(OutputDirectoryError, match="invalid"):
        build_output_layout(root, overwrite=True)
    assert marker.read_text(encoding="utf-8") == "not json"


def test_output_path_that_is_a_file_uses_domain_error(tmp_path):
    output = tmp_path / "run"
    output.write_text("not a directory", encoding="utf-8")
    with pytest.raises(OutputDirectoryError, match="cannot create output directory"):
        build_output_layout(output)


def test_output_lock_prevents_concurrent_writer(tmp_path):
    root = tmp_path / "run"
    with build_output_layout(root):
        with pytest.raises(OutputDirectoryError, match="locked"):
            build_output_layout(root, overwrite=True)


def test_failed_context_marks_run_and_releases_lock(tmp_path):
    root = tmp_path / "run"
    with pytest.raises(RuntimeError):
        with build_output_layout(root):
            raise RuntimeError("boom")
    marker = json.loads((root / ".distillwheel-run.json").read_text(encoding="utf-8"))
    assert marker["status"] == "failed"
    assert "RuntimeError" in marker["error"]
    assert not (root / ".distillwheel.lock").exists()


def test_tail_reads_only_requested_lines(tmp_path):
    with build_output_layout(tmp_path / "run") as layout:
        for index in range(100):
            layout.append_raw_log(str(index))
        assert layout.tail_raw_log(3) == "97\n98\n99\n"
