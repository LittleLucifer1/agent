"""Launcher for ms-swift.

Runs ``<venv>/bin/swift {sft|rlhf} --config <yaml>`` (or the equivalent
``python -m swift.cli ...`` if the CLI isn't on PATH).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

from ...core.envspec import EnvSpec
from ...core.ir.recipe import Recipe
from ...core.launcher import Launcher, filter_env


class SwiftCLILauncher(Launcher):
    def __init__(
        self,
        env_spec: EnvSpec,
        config_path: Path,
        recipe: Recipe,
        workdir: Path,
    ):
        self.env_spec = env_spec
        self._config_path = Path(config_path)
        self._recipe = recipe
        self._workdir = Path(workdir)
        self._native_out = self._workdir / "swift_native"

    def prepare_env(self) -> None:
        from ...core.errors import EnvironmentNotReadyError

        if not self.env_spec.is_ready():
            raise EnvironmentNotReadyError(
                f"swift venv not ready at {self.env_spec.venv_path}. "
                "Run `python tools/setup_backend_envs.py swift` first."
            )
        self._native_out.mkdir(parents=True, exist_ok=True)

    def command(self) -> List[str]:
        from .recipe_mapper import swift_subcommand_for

        subcmd, _ = swift_subcommand_for(self._recipe.stage)

        bin_dir = self.env_spec.python_executable.parent
        swift_bin = bin_dir / ("swift.exe" if os.name == "nt" else "swift")
        if swift_bin.exists():
            argv = [str(swift_bin), subcmd, "--config", str(self._config_path)]
        else:
            argv = [
                str(self.env_spec.python_executable),
                "-m", "swift.cli.main", subcmd,
                "--config", str(self._config_path),
            ]
        return argv

    def env(self) -> dict:
        # Honor parallelism: leave NCCL/CUDA/HF vars to the user shell.
        return filter_env()

    def collect_artifacts(self) -> Path:
        return self._native_out
