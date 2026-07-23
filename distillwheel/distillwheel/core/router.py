"""Stage → backend routing.

Resolution order:
    1. ``recipe.backend_hint`` if set
    2. The default mapping below
    3. Any registered adapter whose ``supports(recipe)`` returns True
"""

from __future__ import annotations

from .adapter import BackendAdapter
from .errors import RegistryError, RoutingError
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
        adapter = get_adapter(recipe.backend_hint)
        if not adapter.supports(recipe):
            raise RoutingError(
                f"backend hint {recipe.backend_hint!r} does not support "
                f"stage={recipe.stage!r}; supported={list(adapter.supported_stages)}"
            )
        return adapter

    if recipe.stage in _DEFAULT_ROUTE:
        default_name = _DEFAULT_ROUTE[recipe.stage]
        try:
            adapter = get_adapter(default_name)
        except RegistryError:
            # The default backend is optional; fall through to plugin discovery.
            pass
        else:
            if not adapter.supports(recipe):
                raise RoutingError(
                    f"default backend {default_name!r} does not support "
                    f"stage={recipe.stage!r}"
                )
            return adapter

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
