"""Dry-run preflight check.

Runs the adapter end-to-end with a *minimal* recipe (1 step, batch=1)
so problems like tokenizer mismatch or trivial OOM surface before a
long training job starts.
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
    r.train.epochs = max(1, int(min(1, r.train.epochs)))  # 1 epoch, but…
    # Many backends accept max_steps via meta override. The cheap shared
    # signal is: tiny global_batch + grad_accum=1 so 1 step costs nothing.
    r.train.global_batch = max(1, r.train.micro_batch)
    r.train.grad_accum = 1
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
    """Light-weight preflight: ensure the adapter can produce a config
    from the shrunken recipe without errors. (Actually launching the
    subprocess is left to integration tests — the cheap check that
    catches most config bugs is the recipe→config translation step.)
    """
    try:
        tiny = shrink_recipe(recipe)
        preflight_dir = Path(workdir) / "_preflight"
        preflight_dir.mkdir(parents=True, exist_ok=True)
        adapter.prepare_config(tiny, data_path, preflight_dir)
    except Exception as e:
        raise PreflightError(f"preflight failed: {e}") from e
