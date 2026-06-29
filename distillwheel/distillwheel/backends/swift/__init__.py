"""swift backend package — registers SwiftAdapter on import."""

from ...core.registry import register_adapter
from .adapter import SwiftAdapter

register_adapter(SwiftAdapter)

__all__ = ["SwiftAdapter"]
