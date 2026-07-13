"""Backend environment specification and path resolution.

Backend environments must not depend on the process's working directory at
launch time.  :func:`backend_env_spec` resolves every managed path eagerly and
also supports an exact per-backend Python override for externally managed
environments (for example, Conda).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence


def _absolute(path: str | os.PathLike[str]) -> Path:
    """Return an expanded, normalized absolute path without requiring it to exist."""
    return Path(path).expanduser().resolve()


def venv_python(venv_path: str | os.PathLike[str]) -> Path:
    """Return the platform-specific Python executable inside a virtualenv."""
    root = _absolute(venv_path)
    if sys.platform == "win32":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


@dataclass
class EnvSpec:
    venv_path: Path
    python_executable: Path
    required_packages: List[str] = field(default_factory=list)
    extra_pip_args: List[str] = field(default_factory=list)
    cuda_constraint: Optional[str] = None
    health_check_cmd: Optional[List[str]] = None
    name: str = ""
    managed: bool = True

    def __post_init__(self) -> None:
        self.venv_path = _absolute(self.venv_path)
        self.python_executable = _absolute(self.python_executable)
        self.required_packages = list(self.required_packages)
        self.extra_pip_args = list(self.extra_pip_args)
        if self.health_check_cmd is not None:
            self.health_check_cmd = list(self.health_check_cmd)

    def is_ready(self) -> bool:
        """True when the configured Python is a runnable file."""
        py = self.python_executable
        return py.is_file() and os.access(py, os.X_OK)

    def resolved_health_check_cmd(self) -> Optional[List[str]]:
        """Return the health command with ``python`` bound to this environment."""
        if not self.health_check_cmd:
            return None
        py = str(self.python_executable)
        return [py if part == "python" else str(part) for part in self.health_check_cmd]

    def run_health_check(self, timeout: float = 30.0) -> bool:
        """Run the configured health check with the environment's Python."""
        command = self.resolved_health_check_cmd()
        if not command:
            return self.is_ready()
        if not self.is_ready():
            return False
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
            return proc.returncode == 0
        except (subprocess.TimeoutExpired, OSError, ValueError):
            return False


def backend_env_spec(
    name: str,
    *,
    required_packages: Sequence[str] = (),
    health_check_cmd: Optional[Sequence[str]] = None,
    extra_pip_args: Sequence[str] = (),
    cuda_constraint: Optional[str] = None,
) -> EnvSpec:
    """Resolve one backend's managed or externally supplied environment.

    ``DISTILLWHEEL_<NAME>_PYTHON`` selects an exact absolute interpreter path.
    Otherwise the environment is ``<DISTILLWHEEL_ENV_ROOT>/<name>``; when the
    root variable is absent it defaults to ``<current working directory>/.venvs``.
    All paths are made absolute immediately, before a launcher can change cwd.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("backend environment name must be non-empty")
    normalized_name = name.strip()
    env_key_name = re.sub(r"[^A-Za-z0-9]", "_", normalized_name).upper()
    python_key = f"DISTILLWHEEL_{env_key_name}_PYTHON"
    python_override = os.environ.get(python_key)

    if python_override:
        override_path = Path(python_override).expanduser()
        if not override_path.is_absolute():
            raise ValueError(f"{python_key} must be an absolute Python path: {python_override!r}")
        python_executable = override_path.resolve()
        # An exact Python override may be a venv or a Conda interpreter.  Do not
        # guess its layout; venv_path is informational for diagnostics only.
        venv_path = python_executable.parent
        managed = False
    else:
        root_value = os.environ.get("DISTILLWHEEL_ENV_ROOT")
        root = _absolute(root_value) if root_value else (Path.cwd() / ".venvs").resolve()
        venv_path = root / normalized_name
        python_executable = venv_python(venv_path)
        managed = True

    return EnvSpec(
        name=normalized_name,
        venv_path=venv_path,
        python_executable=python_executable,
        required_packages=list(required_packages),
        extra_pip_args=list(extra_pip_args),
        cuda_constraint=cuda_constraint,
        health_check_cmd=list(health_check_cmd) if health_check_cmd is not None else None,
        managed=managed,
    )


__all__ = ["EnvSpec", "backend_env_spec", "venv_python"]
