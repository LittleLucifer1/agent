"""SwiftAdapter — ms-swift integration.

This module imports its sibling submodules but does NOT import
``swift`` itself. Anything torch/transformers-related lives only in
the subprocess started by :class:`SwiftCLILauncher`.
"""

from __future__ import annotations

from pathlib import Path

from ...core.adapter import BackendAdapter
from ...core.envspec import EnvSpec
from .checkpoint import SwiftCheckpointNormalizer
from .data_writer import write_swift_jsonl
from .launcher import SwiftCLILauncher
from .logparser import SwiftLogParser
from .recipe_mapper import recipe_to_swift_args


class SwiftAdapter(BackendAdapter):
    name = "swift"
    supported_stages = ("sft", "dpo", "kto")
    env_spec = EnvSpec(
        venv_path=Path(".venvs/swift"),
        python_executable=Path(".venvs/swift/bin/python"),
        required_packages=["ms-swift>=2.5", "transformers>=4.43", "peft>=0.11"],
        health_check_cmd=None,
    )

    def prepare_data(self, stream, recipe, workdir):
        out = Path(workdir) / "data.jsonl"
        write_swift_jsonl(stream, recipe, out)
        return out

    def prepare_config(self, recipe, data_path, workdir):
        workdir = Path(workdir)
        cfg_path = workdir / "swift_config.yaml"
        recipe_to_swift_args(recipe, Path(data_path), cfg_path)
        # also keep an authoritative copy of the IR recipe in workdir
        recipe.to_yaml(workdir / "recipe.yaml")
        return cfg_path

    def build_launcher(self, config_path, recipe, workdir):
        return SwiftCLILauncher(
            env_spec=self.env_spec,
            config_path=Path(config_path),
            recipe=recipe,
            workdir=Path(workdir),
        )

    def checkpoint_normalizer(self):
        return SwiftCheckpointNormalizer()

    def log_parser(self):
        return SwiftLogParser()
