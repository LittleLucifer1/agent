import sys
from pathlib import Path

import pytest

from distillwheel.core.envspec import EnvSpec, backend_env_spec, venv_python


def test_managed_backend_path_is_absolute_and_cwd_stable(tmp_path, monkeypatch):
    monkeypatch.delenv("DISTILLWHEEL_SWIFT_PYTHON", raising=False)
    monkeypatch.delenv("DISTILLWHEEL_ENV_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    spec = backend_env_spec("swift")
    assert spec.venv_path == (tmp_path / ".venvs" / "swift").resolve()
    assert spec.python_executable == venv_python(spec.venv_path)
    assert spec.python_executable.is_absolute()

    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    assert spec.python_executable == venv_python(tmp_path / ".venvs" / "swift")


def test_environment_root_is_resolved_immediately(tmp_path, monkeypatch):
    monkeypatch.delenv("DISTILLWHEEL_VERL_PYTHON", raising=False)
    monkeypatch.setenv("DISTILLWHEEL_ENV_ROOT", str(tmp_path / "env-root"))
    spec = backend_env_spec("verl")
    assert spec.venv_path == (tmp_path / "env-root" / "verl").resolve()


def test_exact_python_override_wins(monkeypatch):
    executable = Path(sys.executable).resolve()
    monkeypatch.setenv("DISTILLWHEEL_SWIFT_PYTHON", str(executable))
    monkeypatch.setenv("DISTILLWHEEL_ENV_ROOT", str(executable.parent / "ignored"))
    spec = backend_env_spec("swift", health_check_cmd=["python", "-c", "print('ok')"])
    assert spec.python_executable == executable
    assert spec.managed is False
    assert spec.resolved_health_check_cmd()[0] == str(executable)
    assert spec.resolved_health_check_cmd()[1] == "-I"
    assert spec.run_health_check()


def test_health_check_does_not_inherit_pythonpath(tmp_path, monkeypatch):
    fake_module = tmp_path / "distillwheel_fake_backend_dependency.py"
    fake_module.write_text("IMPORTED_FROM_PARENT = True\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    spec = EnvSpec(
        venv_path=Path(sys.prefix),
        python_executable=Path(sys.executable),
        health_check_cmd=["python", "-c", "import distillwheel_fake_backend_dependency"],
    )
    assert not spec.run_health_check()


def test_relative_python_override_is_rejected(monkeypatch):
    monkeypatch.setenv("DISTILLWHEEL_SWIFT_PYTHON", "relative/python")
    with pytest.raises(ValueError, match="absolute"):
        backend_env_spec("swift")


def test_venv_python_uses_platform_layout(tmp_path):
    expected = (
        tmp_path / "Scripts" / "python.exe"
        if sys.platform == "win32"
        else tmp_path / "bin" / "python"
    )
    assert venv_python(tmp_path) == expected.resolve()


def test_envspec_readiness_requires_file(tmp_path):
    directory = tmp_path / "python"
    directory.mkdir()
    spec = EnvSpec(venv_path=tmp_path, python_executable=directory)
    assert not spec.is_ready()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_health_check_rejects_invalid_timeout(timeout):
    spec = EnvSpec(
        venv_path=Path(sys.prefix),
        python_executable=Path(sys.executable),
        health_check_cmd=["python", "-c", "print('ok')"],
    )
    assert not spec.run_health_check(timeout=timeout)
