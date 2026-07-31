"""prefixguard — catch silent prompt-cache prefix invalidation before it hits your bill."""

from ._lint import lint
from .guard import PrefixGuard
from .report import CachePrefixBroken, Cause, Finding, TurnReport

__version__ = "0.1.0"

__all__ = [
    "PrefixGuard",
    "lint",
    "Cause",
    "Finding",
    "TurnReport",
    "CachePrefixBroken",
    "__version__",
]
