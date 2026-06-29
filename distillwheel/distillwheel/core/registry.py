"""Global adapter registry.

Backends register themselves either by decorating their class with
:func:`register_adapter`, or via setuptools ``entry_points``
(``distillwheel.backends`` group), which :func:`load_entry_points`
discovers and imports lazily.
"""

from __future__ import annotations

from typing import Dict, List, Type

from .adapter import BackendAdapter
from .errors import RegistryError

_REGISTRY: Dict[str, Type[BackendAdapter]] = {}


def register_adapter(cls: Type[BackendAdapter]) -> Type[BackendAdapter]:
    """Class decorator that registers ``cls`` under ``cls.name``."""
    if not getattr(cls, "name", None):
        raise RegistryError(f"adapter {cls.__name__} missing `name` attribute")
    existing = _REGISTRY.get(cls.name)
    if existing is not None and existing is not cls:
        raise RegistryError(
            f"adapter {cls.name!r} already registered "
            f"(existing={existing.__module__}.{existing.__name__})"
        )
    _REGISTRY[cls.name] = cls
    return cls


def get_adapter(name: str) -> BackendAdapter:
    if name not in _REGISTRY:
        raise RegistryError(
            f"adapter {name!r} not registered. known={list(_REGISTRY)}"
        )
    return _REGISTRY[name]()


def list_adapters() -> List[str]:
    return sorted(_REGISTRY)


def unregister_adapter(name: str) -> None:
    """Mostly for tests."""
    _REGISTRY.pop(name, None)


def load_entry_points(group: str = "distillwheel.backends") -> None:
    """Import every backend package declared in the given entry-point group."""
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        return

    try:
        eps = entry_points(group=group)
    except TypeError:  # python <3.10 fallback path
        eps = entry_points().get(group, [])  # type: ignore[attr-defined]

    for ep in eps:
        try:
            ep.load()
        except Exception as e:  # don't let one broken backend kill startup
            import warnings

            warnings.warn(f"failed to load backend entry point {ep.name!r}: {e}")
