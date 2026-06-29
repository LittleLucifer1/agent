"""BackendAdapter — abstract contract every backend implements.

Adapter methods run in the *main* process. Implementations must NOT
import the underlying training framework at module top level — that
import only happens inside the subprocess that the Launcher spawns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Tuple

from .checkpoint import CheckpointNormalizer
from .envspec import EnvSpec
from .ir.recipe import Recipe
from .ir.sample import SampleStream
from .launcher import Launcher
from .logparser import LogParser


class BackendAdapter(ABC):
    """Single entry point for a training framework.

    Subclasses set ``name``, ``supported_stages`` and ``env_spec`` as
    class attributes, and implement the five abstract methods.
    """

    name: ClassVar[str] = ""
    supported_stages: ClassVar[Tuple[str, ...]] = ()
    env_spec: ClassVar[EnvSpec]

    @abstractmethod
    def prepare_data(self, stream: SampleStream, recipe: Recipe, workdir: Path) -> Path:
        """Stream IR samples into the framework's native data format.

        Returns the path of the produced file (or directory).
        """

    @abstractmethod
    def prepare_config(self, recipe: Recipe, data_path: Path, workdir: Path) -> Path:
        """Translate the IR Recipe into the framework's native config.

        Implementations must also dump the original recipe YAML to
        ``workdir/recipe.yaml`` for reproducibility.
        """

    @abstractmethod
    def build_launcher(self, config_path: Path, recipe: Recipe, workdir: Path) -> Launcher:
        ...

    @abstractmethod
    def checkpoint_normalizer(self) -> CheckpointNormalizer:
        ...

    @abstractmethod
    def log_parser(self) -> LogParser:
        ...

    # ---------- defaults ----------

    def supports(self, recipe: Recipe) -> bool:
        return recipe.stage in self.supported_stages

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} name={self.name!r} stages={list(self.supported_stages)}>"
