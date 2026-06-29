"""Launcher base class + subprocess-isolated default implementation.

All backends launch training as a **separate subprocess** that uses the
backend's own venv python. This is the single most important invariant
of the framework — it prevents version conflicts, crash contagion, and
GPU-state pollution between the main process and the framework code.
"""

from __future__ import annotations

import os
import select
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

from .envspec import EnvSpec
from .errors import EnvironmentNotReadyError, HangDetectedError

# Environment-variable allow-list. Anything not in this set is dropped
# before the subprocess is started, so the parent's PYTHONPATH /
# LD_LIBRARY_PATH cannot leak into the backend venv.
_DEFAULT_ENV_WHITELIST = (
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "TEMP",
    "TMP",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "NVIDIA_VISIBLE_DEVICES",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "HUGGINGFACE_HUB_TOKEN",
    "HF_TOKEN",
    "WANDB_API_KEY",
    "WANDB_PROJECT",
    "WANDB_ENTITY",
    "WANDB_MODE",
    "WANDB_DIR",
    "NCCL_DEBUG",
    "NCCL_SOCKET_IFNAME",
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "RAY_ADDRESS",
    "SYSTEMROOT",     # Windows
    "WINDIR",         # Windows
    "USERPROFILE",    # Windows
    "USERNAME",       # Windows
)


@dataclass
class LaunchResult:
    returncode: int
    artifacts_dir: Path
    duration_s: float
    last_error: Optional[str] = None


def filter_env(
    base_env: Optional[dict] = None,
    whitelist: Optional[tuple] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Return a subset of environment vars that's safe to pass to backends."""
    base = base_env if base_env is not None else os.environ
    allowed = whitelist if whitelist is not None else _DEFAULT_ENV_WHITELIST
    out = {k: v for k, v in base.items() if k in allowed}
    if extra:
        out.update(extra)
    return out


class Launcher(ABC):
    """Subprocess-based training launcher.

    Subclasses must implement :meth:`prepare_env`, :meth:`command`,
    :meth:`env`, and :meth:`collect_artifacts`. The default
    :meth:`launch` is generally good enough — it spawns the subprocess,
    pumps stdout line-by-line, and tracks the exit code.
    """

    env_spec: EnvSpec

    # set by launch()
    _returncode: int = -1
    _start_ts: float = 0.0
    _end_ts: float = 0.0

    # ---------- abstract API ----------

    @abstractmethod
    def prepare_env(self) -> None:
        """Validate the backend venv and prepare workdir / env vars."""

    @abstractmethod
    def command(self) -> List[str]:
        """Full argv (including the venv python/CLI) for the subprocess."""

    @abstractmethod
    def env(self) -> dict:
        """Environment dict to hand to the subprocess.

        Implementations should call :func:`filter_env` to enforce the
        whitelist rather than handing through ``os.environ`` wholesale.
        """

    @abstractmethod
    def collect_artifacts(self) -> Path:
        """Return the framework-native output directory after training."""

    # ---------- default behavior ----------

    def launch(self, *, heartbeat_timeout_s: Optional[float] = None) -> Iterator[str]:
        """Spawn the subprocess and yield stdout lines.

        ``heartbeat_timeout_s`` is an optional watchdog: if no new line
        is produced within this window, the subprocess is killed and a
        :class:`HangDetectedError` is raised after the iterator finishes.
        Defaults to no timeout (long evaluations are legitimate).
        """
        self._start_ts = time.time()
        proc = subprocess.Popen(
            self.command(),
            env=self.env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=self._cwd(),
        )
        assert proc.stdout is not None

        hang_event = threading.Event()
        if heartbeat_timeout_s and heartbeat_timeout_s > 0:
            yield from self._launch_with_heartbeat(proc, heartbeat_timeout_s, hang_event)
        else:
            try:
                for line in proc.stdout:
                    yield line.rstrip("\n")
            finally:
                proc.wait()
                self._returncode = proc.returncode
                self._end_ts = time.time()

        if hang_event.is_set():
            raise HangDetectedError(
                returncode=self._returncode,
                message=f"no stdout for >{heartbeat_timeout_s}s; killed subprocess",
            )

    def _launch_with_heartbeat(
        self,
        proc: "subprocess.Popen[str]",
        timeout_s: float,
        hang_event: threading.Event,
    ) -> Iterator[str]:
        # Cross-platform watchdog: a background thread polls `last_line_ts`
        # and kills the proc if the gap exceeds the threshold.
        state = {"last": time.time()}
        stop = threading.Event()

        def watchdog():
            while not stop.is_set():
                if time.time() - state["last"] > timeout_s and proc.poll() is None:
                    hang_event.set()
                    try:
                        proc.kill()
                    except OSError:
                        pass
                    return
                time.sleep(min(5.0, timeout_s / 4))

        t = threading.Thread(target=watchdog, daemon=True)
        t.start()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                state["last"] = time.time()
                yield line.rstrip("\n")
        finally:
            stop.set()
            proc.wait()
            self._returncode = proc.returncode
            self._end_ts = time.time()

    def _cwd(self) -> Optional[str]:
        """Override if the subprocess needs a specific working directory."""
        return None

    @property
    def returncode(self) -> int:
        return self._returncode

    @property
    def duration_s(self) -> float:
        if self._end_ts and self._start_ts:
            return self._end_ts - self._start_ts
        return 0.0


class SubprocessLauncher(Launcher):
    """Generic subprocess launcher — useful for tests and trivial backends.

    Takes an explicit argv and env dict. Backend adapters typically use
    a subclass that knows how to assemble the argv from their config.
    """

    def __init__(
        self,
        env_spec: EnvSpec,
        argv: List[str],
        artifacts_dir: Path,
        *,
        extra_env: Optional[dict] = None,
        cwd: Optional[Path] = None,
        skip_env_check: bool = False,
    ):
        self.env_spec = env_spec
        self._argv = list(argv)
        self._artifacts_dir = Path(artifacts_dir)
        self._extra_env = dict(extra_env or {})
        self._cwd_path = Path(cwd) if cwd else None
        self._skip_env_check = skip_env_check

    def prepare_env(self) -> None:
        if not self._skip_env_check and not self.env_spec.is_ready():
            raise EnvironmentNotReadyError(
                f"backend venv not ready at {self.env_spec.venv_path}; "
                f"run `tools/setup_backend_envs.py` first."
            )
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)

    def command(self) -> List[str]:
        return self._argv

    def env(self) -> dict:
        return filter_env(extra=self._extra_env)

    def collect_artifacts(self) -> Path:
        return self._artifacts_dir

    def _cwd(self) -> Optional[str]:
        return str(self._cwd_path) if self._cwd_path else None
