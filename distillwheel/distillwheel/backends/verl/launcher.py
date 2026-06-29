"""Ray-job launcher for verl.

Two execution modes (auto-selected by env):

1. **Standalone**: ``python -m verl.trainer.main_ppo +overrides=...``
   when ``RAY_ADDRESS`` is unset.
2. **Cluster**: ``ray job submit --address $RAY_ADDRESS -- python -m verl...``
   when a Ray cluster is already running.
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


# verl trainer entry point per algorithm.
_ENTRY_MODULE = {
    "ppo": "verl.trainer.main_ppo",
    "grpo": "verl.trainer.main_ppo",   # GRPO is invoked through the PPO trainer
    "rloo": "verl.trainer.main_ppo",
    "opd": "verl.trainer.main_ppo",
}


class VerlRayLauncher(Launcher):
    def __init__(
        self,
        env_spec: EnvSpec,
        overrides_path: Path,
        recipe: Recipe,
        workdir: Path,
    ):
        self.env_spec = env_spec
        self._overrides_path = Path(overrides_path)
        self._recipe = recipe
        self._workdir = Path(workdir)
        self._native_out = self._workdir / "verl_native"

    def prepare_env(self) -> None:
        if not self.env_spec.is_ready():
            raise EnvironmentNotReadyError(
                f"verl venv not ready at {self.env_spec.venv_path}. "
                "Run `python tools/setup_backend_envs.py verl` first."
            )
        self._native_out.mkdir(parents=True, exist_ok=True)

    def command(self) -> List[str]:
        algo = verl_algorithm_for(self._recipe.stage)
        entry = _ENTRY_MODULE[algo]
        overrides = load_overrides(self._overrides_path)

        py = str(self.env_spec.python_executable)
        ray_address = os.environ.get("RAY_ADDRESS")
        if ray_address:
            argv = [
                "ray", "job", "submit",
                "--address", ray_address,
                "--working-dir", str(self._workdir),
                "--",
                py, "-m", entry,
            ]
        else:
            argv = [py, "-m", entry]
        argv.extend(overrides)
        return argv

    def env(self) -> dict:
        extra = {
            # ensure verl writes into our managed workdir
            "VERL_OUTPUT_DIR": str(self._native_out),
        }
        return filter_env(extra=extra)

    def collect_artifacts(self) -> Path:
        return self._native_out

    def _cwd(self) -> str:
        return str(self._workdir)
