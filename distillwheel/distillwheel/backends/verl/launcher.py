"""Direct subprocess launcher for VERL 0.8.

VERL's ``main_ppo`` initialises or connects to Ray itself.  Wrapping it in
``ray job submit`` changes the runtime environment and breaks local reward-file
and interpreter assumptions, so DistillWheel always launches the resolved VERL
Python directly.  When ``RAY_ADDRESS`` is present it is passed through and the
VERL/Ray process connects to that cluster; the workdir and backend environment
therefore need to be available on every worker node.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from ...core.envspec import EnvSpec
from ...core.errors import EnvironmentNotReadyError
from ...core.ir.recipe import Recipe
from ...core.launcher import Launcher, filter_env
from .recipe_mapper import load_overrides, verl_algorithm_for


_INHERITED_DISTRIBUTED_KEYS = (
    "NPROC_PER_NODE",
    "NNODES",
    "NODE_RANK",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)


class VerlRayLauncher(Launcher):
    def __init__(
        self,
        env_spec: EnvSpec,
        overrides_path: Path,
        recipe: Recipe,
        workdir: Path,
    ):
        self.env_spec = env_spec
        self._overrides_path = Path(overrides_path).expanduser().resolve()
        self._recipe = recipe
        self._workdir = Path(workdir).expanduser().resolve()
        self._native_out = self._workdir / "verl_native"

    def prepare_env(self) -> None:
        if not self.env_spec.run_health_check():
            raise EnvironmentNotReadyError(
                f"VERL 0.8 environment is not ready at {self.env_spec.python_executable}. "
                "Set DISTILLWHEEL_VERL_PYTHON to an absolute interpreter or "
                "run `python tools/setup_backend_envs.py verl`."
            )
        if not self._overrides_path.is_file():
            raise EnvironmentNotReadyError(f"VERL override file is missing: {self._overrides_path}")
        self._workdir.mkdir(parents=True, exist_ok=True)
        self._native_out.mkdir(parents=True, exist_ok=True)

    def command(self) -> List[str]:
        verl_algorithm_for(self._recipe.stage)  # Validate before spawning.
        python = self.env_spec.python_executable.expanduser().resolve()
        return [
            str(python),
            "-m",
            "verl.trainer.main_ppo",
            *load_overrides(self._overrides_path),
        ]

    def env(self) -> dict:
        # Ray assigns rank and rendezvous variables to its workers.  Values
        # inherited from a parent torchrun/SLURM shell can otherwise leak into
        # the VERL driver and make workers join an unrelated process group.
        base_env = dict(os.environ)
        for key in _INHERITED_DISTRIBUTED_KEYS:
            base_env.pop(key, None)
        return filter_env(base_env=base_env, extra={
            "PYTHONUNBUFFERED": "1",
            "VERL_OUTPUT_DIR": str(self._native_out),
        })

    def collect_artifacts(self) -> Path:
        return self._native_out

    def _cwd(self) -> str:
        return str(self._workdir)
