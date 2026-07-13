"""Unified exception hierarchy for DistillWheel.

All errors raised from the pipeline/adapter layer should derive from
``DistillWheelError`` so callers can catch a single base class.
"""

from __future__ import annotations


class DistillWheelError(Exception):
    """Base class for all DistillWheel errors."""


class IRValidationError(DistillWheelError):
    """Raised when a Sample / Recipe fails schema validation."""


class RegistryError(DistillWheelError):
    """Raised when adapter registration / lookup fails."""


class RoutingError(DistillWheelError):
    """No adapter could be resolved for the given recipe."""


class EnvironmentNotReadyError(DistillWheelError):
    """The backend's isolated venv is missing or unhealthy."""


class TrainingFailedError(DistillWheelError):
    """Subprocess training run exited with a non-zero status.

    The last N lines of stdout are embedded in the message so the caller
    does not have to reach back into the raw logs.
    """

    def __init__(self, returncode: int, tail: str = "", message: str = ""):
        self.returncode = returncode
        self.tail = tail
        msg = message or f"training subprocess exited rc={returncode}"
        if tail:
            msg += f"\n--- last log lines ---\n{tail}"
        super().__init__(msg)


class CheckpointError(DistillWheelError):
    """Raised when checkpoint normalization fails."""


class PreflightError(DistillWheelError):
    """Raised when the dry-run preflight check fails."""


class OutputDirectoryError(DistillWheelError):
    """The requested output directory is unsafe, busy, or already populated."""


class HangDetectedError(TrainingFailedError):
    """Raised when no stdout has been observed for the configured timeout."""
