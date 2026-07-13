"""SubprocessLauncher smoke test using ``echo`` (cross-platform via python -c)."""

import sys
import time
from pathlib import Path

import pytest

from distillwheel.core.envspec import EnvSpec
from distillwheel.core.errors import HangDetectedError
from distillwheel.core.launcher import SubprocessLauncher, filter_env


def test_subprocess_launcher_yields_stdout(tmp_path):
    spec = EnvSpec(
        venv_path=Path(sys.prefix),
        python_executable=Path(sys.executable),
    )
    launcher = SubprocessLauncher(
        env_spec=spec,
        argv=[sys.executable, "-c", "print('hello'); print('step=1 loss=0.5')"],
        artifacts_dir=tmp_path,
    )
    launcher.prepare_env()
    lines = list(launcher.launch())
    assert "hello" in lines
    assert any("step=1" in ln for ln in lines)
    assert launcher.returncode == 0


def test_env_filter_drops_unknown_vars():
    e = filter_env(base_env={
        "PATH": "/x",
        "HF_HOME": "/h",
        "LD_LIBRARY_PATH": "/cuda/lib64",
        "PYTHONPATH": "/must/not/leak",
        "EVIL": "yes",
    })
    assert "PATH" in e
    assert "HF_HOME" in e
    assert e["LD_LIBRARY_PATH"] == "/cuda/lib64"
    assert "PYTHONPATH" not in e
    assert "EVIL" not in e


def _tree_launcher(tmp_path, sentinel):
    child = (
        "import pathlib,sys,time; "
        "time.sleep(1.5); pathlib.Path(sys.argv[1]).write_text('orphan', encoding='utf-8')"
    )
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}, sys.argv[1]]); "
        "print('ready', flush=True); time.sleep(10)"
    )
    spec = EnvSpec(venv_path=Path(sys.prefix), python_executable=Path(sys.executable))
    return SubprocessLauncher(
        env_spec=spec,
        argv=[sys.executable, "-c", parent, str(sentinel)],
        artifacts_dir=tmp_path,
    )


def test_heartbeat_kills_descendant_processes(tmp_path):
    sentinel = tmp_path / "orphan.txt"
    launcher = _tree_launcher(tmp_path, sentinel)
    launcher.prepare_env()
    started = time.monotonic()
    with pytest.raises(HangDetectedError):
        list(launcher.launch(heartbeat_timeout_s=0.2))
    assert time.monotonic() - started < 1.4
    time.sleep(1.6)
    assert not sentinel.exists()


def test_closing_output_iterator_kills_descendants(tmp_path):
    sentinel = tmp_path / "orphan-close.txt"
    launcher = _tree_launcher(tmp_path, sentinel)
    launcher.prepare_env()
    lines = launcher.launch()
    assert next(lines) == "ready"
    started = time.monotonic()
    lines.close()
    assert time.monotonic() - started < 1.4
    time.sleep(1.6)
    assert not sentinel.exists()


def test_subprocess_output_uses_utf8_replacement(tmp_path):
    spec = EnvSpec(venv_path=Path(sys.prefix), python_executable=Path(sys.executable))
    launcher = SubprocessLauncher(
        env_spec=spec,
        argv=[sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'bad: \\xff\\n')"],
        artifacts_dir=tmp_path,
    )
    launcher.prepare_env()
    assert list(launcher.launch()) == ["bad: �"]
