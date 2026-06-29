"""Pipeline package — orchestrator, artifacts layout, preflight."""

from .orchestrator import run_training
from .artifacts import OutputLayout, build_output_layout

__all__ = ["run_training", "OutputLayout", "build_output_layout"]
