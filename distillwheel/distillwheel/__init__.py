"""DistillWheel — training-backend adapter framework.

The package exposes IR types and the pipeline orchestrator at the top level.
Importing this module must NOT pull in any training framework (torch, swift,
verl, ...). Framework code only lives in backend subprocesses.
"""

from .version import __version__

__all__ = ["__version__"]
