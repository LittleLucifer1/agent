from pathlib import Path
from types import SimpleNamespace

from distillwheel.tools import setup_backend_envs as setup


def test_python_probe_uses_isolated_mode(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0, stdout="3.10\n", stderr="")

    monkeypatch.setattr(setup.subprocess, "run", fake_run)

    assert setup._probe_python(["python3.10"]) == "3.10"
    assert captured["command"][:3] == ["python3.10", "-I", "-c"]


def test_create_venv_uses_isolated_mode(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(
        setup.subprocess,
        "run",
        lambda command, **kwargs: captured.append((command, kwargs)),
    )

    target = tmp_path / "backend"
    setup.create_venv(["py", "-3.10"], target, dry_run=False)

    assert captured[0][0] == ["py", "-3.10", "-I", "-m", "venv", str(target)]
    assert captured[0][1]["check"] is True


def test_pip_install_uses_backend_python_in_isolated_mode(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(
        setup.subprocess,
        "run",
        lambda command, **kwargs: captured.append((command, kwargs)),
    )
    python = Path(tmp_path / "python")

    setup.pip_install(
        python,
        ["example==1"],
        ["--extra-index-url", "https://packages.invalid/simple"],
        dry_run=False,
    )

    assert captured[0][0][:5] == [str(python), "-I", "-m", "pip", "install"]
    assert captured[1][0][:5] == [str(python), "-I", "-m", "pip", "install"]
    assert captured[1][0][-3:] == [
        "--extra-index-url",
        "https://packages.invalid/simple",
        "example==1",
    ]
    assert all(kwargs["check"] is True for _, kwargs in captured)
