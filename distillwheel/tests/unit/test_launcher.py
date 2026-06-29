"""SubprocessLauncher smoke test using ``echo`` (cross-platform via python -c)."""

import sys
from pathlib import Path

import pytest

from distillwheel.core.envspec import EnvSpec
from distillwheel.core.launcher import SubprocessLauncher


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
    from distillwheel.core.launcher import filter_env

    e = filter_env(base_env={"PATH": "/x", "HF_HOME": "/h", "EVIL": "yes"})
    assert "PATH" in e
    assert "HF_HOME" in e
    assert "EVIL" not in e
