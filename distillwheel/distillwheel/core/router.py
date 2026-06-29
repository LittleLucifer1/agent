"""Stage → backend routing.

Resolution order:
    1. ``recipe.backend_hint`` if set
    2. The default mapping below
    3. Any registered adapter whose ``supports(recipe)`` returns True
"""

from __future__ import annotations

from .adapter import BackendAdapter
from .errors import RoutingError
from .ir.recipe import Recipe
from .registry import get_adapter, list_adapters

# Default routing. Users override with ``recipe.backend_hint``.
_DEFAULT_ROUTE = {
    "sft":  "swift",
    "dpo":  "swift",
    "kto":  "swift",
    "grpo": "verl",
    "ppo":  "verl",
    "rloo": "verl",
    "opd":  "verl",
}


def resolve(recipe: Recipe) -> BackendAdapter:
    if recipe.backend_hint:
        return get_adapter(recipe.backend_hint)

    if recipe.stage in _DEFAULT_ROUTE:
        try:
            return get_adapter(_DEFAULT_ROUTE[recipe.stage])
        except Exception:
            # default route adapter not registered; fall through to discovery
            pass

    for name in list_adapters():
        ad = get_adapter(name)
        if ad.supports(recipe):
            return ad

    raise RoutingError(
        f"no adapter supports stage={recipe.stage!r}. "
        f"registered={list_adapters()}"
    )


def default_route_for(stage: str) -> str:
    return _DEFAULT_ROUTE.get(stage, "")
