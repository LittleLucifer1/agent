"""verl backend package — registers VerlAdapter on import."""

from ...core.registry import register_adapter
from .adapter import VerlAdapter

register_adapter(VerlAdapter)

__all__ = ["VerlAdapter"]
