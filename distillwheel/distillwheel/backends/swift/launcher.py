"""Subprocess launcher for ms-swift 4.4.

ms-swift's YAML parser consumes the config as the first positional argument,
for example ``swift sft config.yaml``.  The console script delegates to the
same ``swift.cli.main`` module used by the fallback below.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from ...core.envspec import EnvSpec
from ...core.errors import IRValidationError
from ...core.ir.recipe import Recipe
from ...core.launcher import Launcher, filter_env
from .recipe_mapper import swift_subcommand_for


_INHERITED_DISTRIBUTED_KEYS = (
    "NPROC_PER_NODE",
    "NNODES",
    "NODE_RANK",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
)


class SwiftCLILauncher(Launcher):
    def __init__(
        self,
        env_spec: EnvSpec,
        config_path: Path,
        recipe: Recipe,
        workdir: Path,
    ):
        self.env_spec = env_spec
        self._config_path = Path(config_path).expanduser().resolve()
        self._recipe = recipe
        self._workdir = Path(workdir).expanduser().resolve()
        self._native_out = self._workdir / "swift_native"

    def prepare_env(self) -> None:
        from ...core.errors import EnvironmentNotReadyError

        if not self.env_spec.run_health_check():
            raise EnvironmentNotReadyError(
                f"swift 4.4 environment not ready at "
                f"{self.env_spec.python_executable}. Run "
                "`python tools/setup_backend_envs.py swift` first."
            )
        self._native_out.mkdir(parents=True, exist_ok=True)

    def command(self) -> List[str]:
        subcmd, _ = swift_subcommand_for(self._recipe.stage)

        # setup.py in ms-swift 4.4 registers swift=swift.cli.main:cli_main.
        # If that console script is absent, invoking the same module via the
        # environment's absolute Python path is equivalent and supported by
        # its __main__ guard.
        bin_dir = self.env_spec.python_executable.parent
        swift_bin = bin_dir / ("swift.exe" if os.name == "nt" else "swift")
        if swift_bin.is_file() and os.access(swift_bin, os.X_OK):
            return [str(swift_bin.resolve()), subcmd, str(self._config_path)]
        return [
            str(self.env_spec.python_executable.resolve()),
            "-m",
            "swift.cli.main",
            subcmd,
            str(self._config_path),
        ]

    def env(self) -> dict:
        dp = self._recipe.parallel.dp
        if not isinstance(dp, int) or isinstance(dp, bool) or dp < 1:
            raise ValueError("recipe.parallel.dp must be a positive integer")

        nnodes_raw = os.environ.get("NNODES")
        if nnodes_raw is not None:
            try:
                nnodes = int(nnodes_raw)
            except ValueError as exc:
                raise IRValidationError(
                    "NNODES must be an integer; multi-node Swift runs require "
                    "a future ResourceConfig extension"
                ) from exc
            if nnodes != 1:
                raise IRValidationError(
                    "the current Recipe IR defines only single-node "
                    "parallel.dp; NNODES>1 requires a future ResourceConfig "
                    "extension so global data parallelism is not multiplied"
                )

        # The recipe is authoritative.  Inheriting launcher state from a
        # parent torchrun can multiply world size or reuse an invalid rank.
        base_env = dict(os.environ)
        for key in _INHERITED_DISTRIBUTED_KEYS:
            base_env.pop(key, None)
        extra = {"NPROC_PER_NODE": str(dp)} if dp > 1 else None
        return filter_env(base_env=base_env, extra=extra)

    def collect_artifacts(self) -> Path:
        return self._native_out
