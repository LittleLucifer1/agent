"""Cheap configuration-level preflight check.

This check only asks the adapter to translate a one-step Recipe. It does not
load a model, tokenizer, or launch the training framework, so runtime and OOM
failures remain the responsibility of backend integration tests or a real run.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

from ..core.adapter import BackendAdapter
from ..core.errors import PreflightError
from ..core.ir.recipe import Recipe


def shrink_recipe(recipe: Recipe) -> Recipe:
    """Return a deep-copied recipe with the train budget cut to 1 micro-step."""
    r = copy.deepcopy(recipe)
    r.train.epochs = 1.0
    r.train.micro_batch = 1
    r.train.grad_accum = 1
    r.train.global_batch = r.parallel.dp
    r.io.save_steps = 10_000_000  # don't save anything during preflight
    r.io.logging_steps = 1
    r.meta = dict(r.meta or {})
    r.meta["__preflight__"] = True
    r.meta["max_steps"] = 1
    return r


def run_preflight(
    adapter: BackendAdapter,
    recipe: Recipe,
    data_path: Path,
    workdir: Path,
) -> None:
    """Ensure the adapter can translate a validated, one-step Recipe."""
    try:
        tiny = shrink_recipe(recipe)
        tiny.validate()
        preflight_dir = Path(workdir) / "_preflight"
        preflight_dir.mkdir(parents=True, exist_ok=True)
        adapter.prepare_config(tiny, data_path, preflight_dir)
    except Exception as e:
        raise PreflightError(f"preflight failed: {e}") from e
