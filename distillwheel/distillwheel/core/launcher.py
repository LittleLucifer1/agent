<<<<<<< HEAD
"""Launcher base class and subprocess-isolated default implementation."""
=======
"""Launcher base class + subprocess-isolated default implementation.

All backends launch training as a **separate subprocess** that uses the
backend's own venv python. This is the single most important invariant
of the framework — it prevents version conflicts, crash contagion, and
GPU-state pollution between the main process and the framework code.

子进程隔离三件套
================

1. **独立 venv** — 每个 backend 用自己的 venv python, 不共享主进程依赖
2. **环境变量白名单** — 主进程的 PYTHONPATH / LD_LIBRARY_PATH 不会泄漏到子进程,
   只透传 CUDA / HF / WANDB / NCCL 等必要变量
3. **心跳超时** — 后台 watchdog 线程监控 stdout, 长时间无输出则怀疑 NCCL hang 并杀进程

执行流程
=======
    Orchestrator
        ▼
    launcher.prepare_env()          # 1) 检查 venv 就绪, 创建工作目录
        │
        ▼
    launcher.launch()               # 2) subprocess.Popen 启动训练
        │                                命令行 = launcher.command()
        │                                环境变量 = launcher.env()  ← filter_env 白名单过滤
        │
        ├──▶ yield stdout line      # 3) 逐行 yield 给 orchestrator
        │       │                        orchestrator 同时做:
        │       ├─ 写入 raw_logs/        - 原始日志落盘
        │       └─ LogParser.parse_line  - 抽取归一化指标
        ▼
    launcher.collect_artifacts()    # 4) 训练结束, 返回框架原生输出目录 交给 CheckpointNormalizer 处理
"""
>>>>>>> 67e62d2cdb9b79eef20cf4f3e7a520c5a169609a

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

from .envspec import EnvSpec
from .errors import EnvironmentNotReadyError, HangDetectedError

<<<<<<< HEAD
# Environment-variable allow-list. Anything not in this set is dropped before
# the subprocess is started, so the parent's PYTHONPATH cannot leak into the
# backend environment.
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
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_MODULE_LOADING",
    "NVIDIA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "ROCM_HOME",
    "HIP_PATH",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_DATASETS_CACHE",
    "HF_ENDPOINT",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_CACHE",
    "TRANSFORMERS_OFFLINE",
    "HUGGINGFACE_HUB_TOKEN",
    "HF_TOKEN",
    "TORCH_HOME",
    "TRITON_CACHE_DIR",
    "XDG_CACHE_HOME",
    "MODELSCOPE_CACHE",
    "MODELSCOPE_DOMAIN",
    "MODELSCOPE_OFFLINE",
    "USE_HF",
    "VERL_USE_MODELSCOPE",
    "WANDB_API_KEY",
    "WANDB_PROJECT",
    "WANDB_ENTITY",
    "WANDB_MODE",
    "WANDB_DIR",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "NCCL_SOCKET_IFNAME",
    "NCCL_SOCKET_NTHREADS",
    "NCCL_NSOCKS_PERTHREAD",
    "NCCL_IB_DISABLE",
    "NCCL_IB_HCA",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_TIMEOUT",
    "NCCL_IB_RETRY_CNT",
    "NCCL_P2P_DISABLE",
    "NCCL_P2P_LEVEL",
    "NCCL_SHM_DISABLE",
    "NCCL_NET_GDR_LEVEL",
    "NCCL_CROSS_NIC",
    "NCCL_ASYNC_ERROR_HANDLING",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TOKENIZERS_PARALLELISM",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "NPROC_PER_NODE",
    "NNODES",
    "NODE_RANK",
    "RAY_ADDRESS",
    "RAY_DEDUP_LOGS",
    "GLOO_SOCKET_IFNAME",
    "HYDRA_FULL_ERROR",
    "VLLM_USE_V1",
    "PYTHONUNBUFFERED",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "SYSTEMROOT",     # Windows
    "WINDIR",         # Windows
    "USERPROFILE",    # Windows
    "USERNAME",       # Windows
=======
# ── 环境变量白名单 ──────────────────────────────────────────────────
# 只有这些变量会被透传给训练子进程, 其他全部丢弃。
# 这样主进程的 PYTHONPATH / LD_LIBRARY_PATH 不会污染 backend 的 venv。
_DEFAULT_ENV_WHITELIST = (
    # 基础系统
    "PATH", "HOME", "USER", "LANG", "LC_ALL",
    "TMPDIR", "TEMP", "TMP",
    # GPU / CUDA
    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NVIDIA_VISIBLE_DEVICES",
    # HuggingFace
    "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
    "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN",
    # Weights & Biases
    "WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_MODE", "WANDB_DIR",
    # 分布式训练 (NCCL / torch.distributed)
    "NCCL_DEBUG", "NCCL_SOCKET_IFNAME",
    "MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK",
    # Ray (verl backend)
    "RAY_ADDRESS",
    # Windows
    "SYSTEMROOT", "WINDIR", "USERPROFILE", "USERNAME",
>>>>>>> 67e62d2cdb9b79eef20cf4f3e7a520c5a169609a
)


@dataclass
class LaunchResult:
    """训练子进程的执行结果 (目前未直接使用, 保留供未来异步场景)。"""
    returncode: int                              # 子进程退出码, 0 = 成功
    artifacts_dir: Path                          # 框架原生输出目录
    duration_s: float                            # 训练耗时 (秒)
    last_error: Optional[str] = None             # 最后 N 行 stderr (用于诊断)


def filter_env(
    base_env: Optional[dict] = None,
    whitelist: Optional[tuple] = None,
    extra: Optional[dict] = None,
) -> dict:
<<<<<<< HEAD
    """Return a subset of environment variables safe for backend processes."""
=======
    """过滤环境变量, 只保留白名单中的键。

    所有 Launcher 子类的 ``env()`` 方法都应通过此函数构建环境变量,
    而不是直接传 ``os.environ`` — 否则 PYTHONPATH 等变量会污染 backend venv。

    Parameters
    ----------
    base_env : dict, optional
        源环境变量字典, 默认为 ``os.environ``
    whitelist : tuple, optional
        允许透传的变量名集合, 默认为 ``_DEFAULT_ENV_WHITELIST``
    extra : dict, optional
        额外注入的变量 (如 ``MOCK_NATIVE_DIR``), 无条件合入
    """
>>>>>>> 67e62d2cdb9b79eef20cf4f3e7a520c5a169609a
    base = base_env if base_env is not None else os.environ
    allowed = whitelist if whitelist is not None else _DEFAULT_ENV_WHITELIST
    out = {str(k): str(v) for k, v in base.items() if k in allowed}
    if extra:
        out.update({str(k): str(v) for k, v in extra.items()})
    return out


# ════════════════════════════════════════════════════════════════════
# Launcher 基类
# ════════════════════════════════════════════════════════════════════

<<<<<<< HEAD
    Every launched backend receives its own process group.  Closing the output
    iterator early, a consumer exception, Ctrl-C, or a heartbeat timeout tears
    down that group before waiting, so framework workers are not orphaned.
=======
class Launcher(ABC):
    """训练子进程的统一接口。

    **所有 backend 必须用子进程方式启动训练**, 禁止 in-process 调用 — 这能避免:

    - 版本冲突 (主进程和 backend 的 torch 版本不同)
    - 崩溃传染 (子进程 segfault 不影响主进程)
    - GPU 状态污染 (子进程退出后 GPU 资源自动释放)

    子类需要实现 4 个抽象方法:

    - ``prepare_env()`` — 检查 venv 就绪、创建工作目录
    - ``command()`` — 返回完整命令行 (用 backend 自己的 venv python)
    - ``env()`` — 返回传给子进程的环境变量 (应调用 ``filter_env()``)
    - ``collect_artifacts()`` — 训练结束后返回框架原生输出目录

    ``launch()`` 的默认实现一般不需要重写。
>>>>>>> 67e62d2cdb9b79eef20cf4f3e7a520c5a169609a
    """

    env_spec: EnvSpec

<<<<<<< HEAD
    _returncode: int = -1
=======
    _returncode: int = -1                        # 由 launch() 设置
>>>>>>> 67e62d2cdb9b79eef20cf4f3e7a520c5a169609a
    _start_ts: float = 0.0
    _end_ts: float = 0.0

    @abstractmethod
    def prepare_env(self) -> None:
<<<<<<< HEAD
        """Validate the backend environment and prepare its work directory."""

    @abstractmethod
    def command(self) -> List[str]:
        """Full argv for the subprocess."""

    @abstractmethod
    def env(self) -> dict:
        """Environment mapping to hand to the subprocess."""
=======
        """检查 backend venv 就绪性, 创建工作目录和环境变量。"""

    @abstractmethod
    def command(self) -> List[str]:
        """返回完整命令行 (含 venv 内的解释器/CLI 路径)。

        示例::

            # swift
            [".venvs/swift/bin/swift", "sft", "--config", "swift_config.yaml"]
            # verl
            [".venvs/verl/bin/python", "-m", "verl.trainer.main_ppo", "algorithm=grpo", ...]
        """

    @abstractmethod
    def env(self) -> dict:
        """返回传给子进程的环境变量字典。

        实现时应调用 ``filter_env()`` 来强制白名单过滤,
        而不是直接传 ``os.environ``。
        """
>>>>>>> 67e62d2cdb9b79eef20cf4f3e7a520c5a169609a

    @abstractmethod
    def collect_artifacts(self) -> Path:
        """训练结束后返回框架原生输出目录, 供 CheckpointNormalizer 处理。"""

    def launch(self, *, heartbeat_timeout_s: Optional[float] = None) -> Iterator[str]:
<<<<<<< HEAD
        """Spawn the subprocess and yield combined stdout/stderr lines."""
        if heartbeat_timeout_s is not None and heartbeat_timeout_s <= 0:
            raise ValueError("heartbeat_timeout_s must be > 0 when provided")

        argv = [str(part) for part in self.command()]
        if not argv:
            raise EnvironmentNotReadyError("backend launcher produced an empty command")
=======
        """启动训练子进程, 逐行 yield stdout。

        这是一个生成器 — orchestrator 在 for 循环中同时完成日志落盘和指标解析。

        Parameters
        ----------
        heartbeat_timeout_s : float, optional
            心跳超时 (秒)。如果子进程连续这么久没有新 stdout, watchdog 线程
            会杀掉子进程并在迭代结束后抛出 :class:`HangDetectedError`。
            典型场景: NCCL 多卡通信 hang 住, 进程卡死不输出。
            默认为 None (不设超时), 因为长时间的 eval/checkpoint 是合法的。
        """
        self._start_ts = time.time()
        proc = subprocess.Popen(
            self.command(),
            env=self.env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,           # stderr 合并到 stdout, 统一处理
            text=True,
            bufsize=1,                           # 行缓冲, 保证实时输出
            cwd=self._cwd(),
        )
        assert proc.stdout is not None
>>>>>>> 67e62d2cdb9b79eef20cf4f3e7a520c5a169609a

        child_env = dict(self.env())
        child_env.setdefault("PYTHONUNBUFFERED", "1")
        popen_group: dict = {}
        if os.name == "nt":
            popen_group["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_group["start_new_session"] = True

        self._returncode = -1
        self._start_ts = time.monotonic()
        self._end_ts = 0.0
        try:
            proc: "subprocess.Popen[str]" = subprocess.Popen(
                argv,
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=self._cwd(),
                **popen_group,
            )
        except (OSError, ValueError) as exc:
            self._end_ts = time.monotonic()
            raise EnvironmentNotReadyError(
                f"failed to start backend command {argv[0]!r}: {exc}"
            ) from exc

        assert proc.stdout is not None
        hang_event = threading.Event()
        watchdog_stop = threading.Event()
        terminate_lock = threading.Lock()
        last_output = [time.monotonic()]
        watchdog: Optional[threading.Thread] = None

        if heartbeat_timeout_s is not None:
            timeout_s = heartbeat_timeout_s

            def watch() -> None:
                interval = max(0.01, min(0.25, timeout_s / 4.0))
                while not watchdog_stop.wait(interval):
                    if proc.poll() is not None:
                        return
                    if time.monotonic() - last_output[0] > timeout_s:
                        hang_event.set()
                        self._terminate_process_tree(proc, terminate_lock)
                        return

            watchdog = threading.Thread(
                target=watch,
                name="distillwheel-heartbeat",
                daemon=True,
            )
            watchdog.start()

        natural_eof = False
        try:
            for line in proc.stdout:
                last_output[0] = time.monotonic()
                yield line.rstrip("\n")
            natural_eof = True
        finally:
            # If the caller closes the generator or raises while consuming a
            # line, do not wait forever for a still-running training tree.
            if not natural_eof and proc.poll() is None:
                self._terminate_process_tree(proc, terminate_lock)
            watchdog_stop.set()
            if watchdog is not None:
                watchdog.join(timeout=1.0)
            try:
                proc.stdout.close()
            except OSError:
                pass
            if proc.poll() is None:
                # Natural EOF normally means the process is just about to exit.
                # Bound the wait; a process that closed stdout but stayed alive
                # is still considered part of the managed tree.
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._terminate_process_tree(proc, terminate_lock)
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                # Last-resort single-process kill; tree helpers have already
                # been attempted on every supported platform.
                try:
                    proc.kill()
                except OSError:
                    pass
                proc.wait()
            self._returncode = int(proc.returncode)
            self._end_ts = time.monotonic()

        if hang_event.is_set():
            raise HangDetectedError(
                returncode=self._returncode,
                message=f"no stdout for >{heartbeat_timeout_s}s; killed subprocess tree",
            )

    @staticmethod
    def _terminate_process_tree(
        proc: "subprocess.Popen[str]",
<<<<<<< HEAD
        lock: Optional[threading.Lock] = None,
        *,
        grace_s: float = 0.5,
    ) -> None:
        """Best-effort cross-platform termination of ``proc`` and descendants."""
        guard = lock or threading.Lock()
        with guard:
            if os.name == "nt":
                Launcher._terminate_windows_tree(proc, grace_s)
            else:
                Launcher._terminate_posix_group(proc, grace_s)

    @staticmethod
    def _terminate_posix_group(proc: "subprocess.Popen[str]", grace_s: float) -> None:
        pgid = proc.pid
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            try:
                proc.terminate()
            except OSError:
                return

        deadline = time.monotonic() + max(0.0, grace_s)
        while time.monotonic() < deadline:
            proc.poll()  # reap the group leader when possible
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return
            except OSError:
                break
            time.sleep(0.02)

        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass

    @staticmethod
    def _terminate_windows_tree(proc: "subprocess.Popen[str]", grace_s: float) -> None:
        # ``CTRL_BREAK_EVENT`` can make the group leader exit before every
        # descendant has handled the event.  Once that happens ``taskkill /T``
        # can no longer discover the tree from the parent PID.  Capture and
        # terminate the tree while the leader is still alive instead.  The
        # force flag is intentional: iterator cancellation and heartbeat
        # expiry require deterministic cleanup of training workers.
        system_root = os.environ.get("SystemRoot") or os.environ.get("SYSTEMROOT")
        taskkill = (
            str(Path(system_root) / "System32" / "taskkill.exe")
            if system_root
            else shutil.which("taskkill")
        )
        if taskkill:
            try:
                subprocess.run(
                    [taskkill, "/PID", str(proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5.0,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass

        # A restricted Windows installation may not provide ``taskkill``.
        # Fall back to the process-group control event, then the leader-only
        # kill exposed by ``subprocess``.
        if proc.poll() is None:
            try:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
                proc.wait(timeout=max(0.0, grace_s))
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass

    def _cwd(self) -> Optional[str]:
=======
        timeout_s: float,
        hang_event: threading.Event,
    ) -> Iterator[str]:
        """带心跳监控的 launch 实现。

        原理: 后台 watchdog 线程定期检查上次收到 stdout 的时间戳,
        如果超过阈值且进程仍在运行, 则 kill 子进程并设置 hang_event。
        """
        state = {"last": time.time()}
        stop = threading.Event()

        def watchdog():
            while not stop.is_set():
                idle = time.time() - state["last"]
                if idle > timeout_s and proc.poll() is None:
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
                state["last"] = time.time()      # 每收到一行就刷新心跳时间戳
                yield line.rstrip("\n")
        finally:
            stop.set()
            proc.wait()
            self._returncode = proc.returncode
            self._end_ts = time.time()

    def _cwd(self) -> Optional[str]:
        """子进程的工作目录。子类可重写, 默认不指定 (继承主进程 cwd)。"""
>>>>>>> 67e62d2cdb9b79eef20cf4f3e7a520c5a169609a
        return None

    @property
    def returncode(self) -> int:
        """子进程退出码。launch() 结束前为 -1。"""
        return self._returncode

    @property
    def duration_s(self) -> float:
        """训练耗时 (秒)。launch() 结束前为 0。"""
        if self._end_ts and self._start_ts:
            return self._end_ts - self._start_ts
        return 0.0


<<<<<<< HEAD
class SubprocessLauncher(Launcher):
    """Generic subprocess launcher useful for tests and simple backends."""
=======
# ════════════════════════════════════════════════════════════════════
# SubprocessLauncher — 通用实现, 主要用于测试和 mock backend
# ════════════════════════════════════════════════════════════════════

class SubprocessLauncher(Launcher):
    """通用子进程 launcher — 接受显式的 argv 和 env。

    真正的 backend (如 SwiftCLILauncher / VerlRayLauncher) 通常继承
    Launcher 基类并自己组装命令行, 而不是用这个类。
    这个类主要服务于:

    - 单元测试 (用 ``echo`` / ``python -c`` 模拟训练)
    - 简单的 wrapper backend
    """
>>>>>>> 67e62d2cdb9b79eef20cf4f3e7a520c5a169609a

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
        self._cwd_path = Path(cwd).resolve() if cwd else None
        self._skip_env_check = skip_env_check

    def prepare_env(self) -> None:
        if not self._skip_env_check and not self.env_spec.is_ready():
            raise EnvironmentNotReadyError(
                f"backend environment not ready at {self.env_spec.python_executable}; "
                "run `distillwheel-setup-backends` first"
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


__all__ = ["LaunchResult", "Launcher", "SubprocessLauncher", "filter_env"]
