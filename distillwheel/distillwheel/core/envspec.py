"""Backend environment specification.

Each adapter declares an :class:`EnvSpec` describing the isolated venv
(or container) it needs. ``tools/setup_backend_envs.py`` consumes these
to create the venvs on first install.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class EnvSpec:
    venv_path: Path
    python_executable: Path
    required_packages: List[str] = field(default_factory=list)
    extra_pip_args: List[str] = field(default_factory=list)
    cuda_constraint: Optional[str] = None
    health_check_cmd: Optional[List[str]] = None

    def is_ready(self) -> bool:
        """True when the venv python binary exists on disk."""
        return Path(self.python_executable).exists()

    def run_health_check(self, timeout: float = 30.0) -> bool:
        """Run ``health_check_cmd`` if configured; return True on success."""
        if not self.health_check_cmd:
            return True
        try:
            proc = subprocess.run(
                self.health_check_cmd,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            return proc.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False
