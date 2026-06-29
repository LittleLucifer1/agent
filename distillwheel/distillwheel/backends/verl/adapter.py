"""VerlAdapter — verl integration for GRPO / PPO / RLOO / OPD.

Like the swift adapter, this module never imports ``verl`` itself in the
main process. The training code only runs inside the launcher's
subprocess, in the verl venv.
"""

from __future__ import annotations

from pathlib import Path

from ...core.adapter import BackendAdapter
from ...core.envspec import EnvSpec
from .checkpoint import VerlCheckpointNormalizer
from .data_writer import write_verl_parquet
from .launcher import VerlRayLauncher
from .logparser import VerlLogParser
from .recipe_mapper import recipe_to_verl_overrides


class VerlAdapter(BackendAdapter):
    name = "verl"
    supported_stages = ("grpo", "ppo", "rloo", "opd")
    env_spec = EnvSpec(
        venv_path=Path(".venvs/verl"),
        python_executable=Path(".venvs/verl/bin/python"),
        required_packages=["verl", "ray", "vllm", "pyarrow"],
    )

    def prepare_data(self, stream, recipe, workdir):
        out = Path(workdir) / "prompts.parquet"
        write_verl_parquet(stream, recipe, out)
        return out

    def prepare_config(self, recipe, data_path, workdir):
        workdir = Path(workdir)
        overrides_path = workdir / "verl_overrides.txt"
        recipe_to_verl_overrides(recipe, Path(data_path), overrides_path)
        recipe.to_yaml(workdir / "recipe.yaml")
        return overrides_path

    def build_launcher(self, config_path, recipe, workdir):
        return VerlRayLauncher(
            env_spec=self.env_spec,
            overrides_path=Path(config_path),
            recipe=recipe,
            workdir=Path(workdir),
        )

    def checkpoint_normalizer(self):
        return VerlCheckpointNormalizer(env_spec=self.env_spec)

    def log_parser(self):
        return VerlLogParser()
