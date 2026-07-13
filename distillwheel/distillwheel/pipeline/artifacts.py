"""Managed output directory layout and run locking."""

from __future__ import annotations

import json
import os
import shutil
import socket
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Optional

from ..core.checkpoint import NormalizedCheckpoint
from ..core.errors import OutputDirectoryError

_MARKER_NAME = ".distillwheel-run.json"
_LOCK_NAME = ".distillwheel.lock"
_MANAGED_NAMES = (
    "workdir",
    "raw_logs",
    "metrics.jsonl",
    "final",
    "checkpoints",
    "training_recipe.yaml",
    "metadata.json",
    _MARKER_NAME,
)


@dataclass
class OutputLayout:
    root: Path
    workdir: Path
    raw_logs_dir: Path
    metrics_jsonl: Path
    final_dir: Path
    checkpoints_dir: Path
    recipe_yaml: Path
    metadata_json: Path
    run_id: str
    _raw_log_path: Path
    _marker_path: Path
    _lock_path: Path
    _lock_file: Optional[IO[str]] = field(default=None, repr=False)
    _closed: bool = field(default=False, repr=False)

    def __enter__(self) -> "OutputLayout":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc is None:
                self._write_run_marker("succeeded")
            else:
                self._write_run_marker(
                    "failed",
                    error=f"{exc_type.__name__}: {exc}"[:2000] if exc_type else str(exc)[:2000],
                )
        finally:
            self.close()

    def append_raw_log(self, line: str) -> None:
        """Append one stdout line to this run's raw log."""
        with open(self._raw_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def tail_raw_log(self, n: int = 50) -> str:
        """Return the last ``n`` log lines without loading the whole file."""
        if n <= 0:
            return ""
        try:
            with open(self._raw_log_path, "r", encoding="utf-8", errors="replace") as f:
                return "".join(deque(f, maxlen=n))
        except OSError:
            return ""

    def write_metadata(self, ck: NormalizedCheckpoint) -> None:
        _atomic_write_json(self.metadata_json, ck.to_dict())

    def close(self) -> None:
        """Release the per-run lock. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        lock_file = self._lock_file
        self._lock_file = None
        if lock_file is not None:
            try:
                lock_file.close()
            finally:
                try:
                    payload = self._lock_path.read_text(encoding="utf-8")
                    if self.run_id in payload:
                        self._lock_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _write_run_marker(self, status: str, *, error: Optional[str] = None) -> None:
        payload = {
            "format_version": 1,
            "run_id": self.run_id,
            "status": status,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "updated_at": time.time(),
        }
        if error:
            payload["error"] = error
        _atomic_write_json(self._marker_path, payload)


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _acquire_lock(root: Path, run_id: str) -> tuple[Path, IO[str]]:
    lock_path = root / _LOCK_NAME
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        details = ""
        try:
            details = f" ({lock_path.read_text(encoding='utf-8').strip()})"
        except OSError:
            pass
        raise OutputDirectoryError(f"output directory is locked: {root}{details}") from exc
    lock_file = os.fdopen(fd, "w", encoding="utf-8")
    lock_file.write(
        json.dumps({
            "run_id": run_id,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "created_at": time.time(),
        }, ensure_ascii=False)
    )
    lock_file.flush()
    os.fsync(lock_file.fileno())
    return lock_path, lock_file


def _remove_managed_outputs(root: Path) -> None:
    for name in _MANAGED_NAMES:
        candidate = root / name
        if not candidate.exists() and not candidate.is_symlink():
            continue
        resolved = candidate.resolve()
        if resolved.parent != root:
            raise OutputDirectoryError(
                f"refusing to remove managed path outside output root: {candidate} -> {resolved}"
            )
        if candidate.is_dir() and not candidate.is_symlink():
            shutil.rmtree(candidate)
        else:
            candidate.unlink()


def build_output_layout(
    output_dir: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> OutputLayout:
    """Prepare and lock a single-run output directory.

    A non-empty directory is rejected by default. ``overwrite=True`` only
    removes known framework-managed paths and is allowed only when a prior
    DistillWheel marker exists. Use the returned object as a context manager so
    the lock is always released and the run marker is finalized.
    """
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    lock_path, lock_file = _acquire_lock(root, run_id)
    marker_path = root / _MARKER_NAME

    try:
        existing = [p for p in root.iterdir() if p.name != _LOCK_NAME]
        if existing:
            if not overwrite:
                raise OutputDirectoryError(
                    f"output directory is not empty: {root}. "
                    "Choose a new directory or pass --overwrite-output."
                )
            if not marker_path.is_file():
                raise OutputDirectoryError(
                    f"refusing to overwrite unmarked directory: {root}. "
                    f"Expected {_MARKER_NAME!r}."
                )
            _remove_managed_outputs(root)

        # Establish ownership before creating managed directories. If setup is
        # interrupted halfway through, a later explicit overwrite can safely
        # recognize and clean the partial run.
        _atomic_write_json(marker_path, {
            "format_version": 1,
            "run_id": run_id,
            "status": "preparing",
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "updated_at": time.time(),
        })

        workdir = root / "workdir"
        raw_logs = root / "raw_logs"
        final_dir = root / "final"
        checkpoints_dir = root / "checkpoints"
        for path in (workdir, raw_logs, final_dir, checkpoints_dir):
            path.mkdir(parents=True, exist_ok=False)

        raw_log_path = raw_logs / "stdout.log"
        raw_log_path.write_text("", encoding="utf-8")
        layout = OutputLayout(
            root=root,
            workdir=workdir,
            raw_logs_dir=raw_logs,
            metrics_jsonl=root / "metrics.jsonl",
            final_dir=final_dir,
            checkpoints_dir=checkpoints_dir,
            recipe_yaml=root / "training_recipe.yaml",
            metadata_json=root / "metadata.json",
            run_id=run_id,
            _raw_log_path=raw_log_path,
            _marker_path=marker_path,
            _lock_path=lock_path,
            _lock_file=lock_file,
        )
        layout._write_run_marker("running")
        return layout
    except Exception:
        try:
            lock_file.close()
        finally:
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


__all__ = ["OutputLayout", "build_output_layout"]
