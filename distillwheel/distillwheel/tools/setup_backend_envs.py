"""Create and validate isolated backend Python environments."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

import yaml

from ..core.envspec import EnvSpec, backend_env_spec

_PYTHON_VERSION_RE = re.compile(r"^\d+\.\d+$")


class BackendSetupError(RuntimeError):
    """A backend environment could not be created or validated."""


def discover_backends() -> dict[str, object]:
    """Return package resources for built-in backend manifests."""
    root = resources.files("distillwheel.backends")
    found: dict[str, object] = {}
    for child in root.iterdir():
        manifest = child.joinpath("env.yaml")
        if child.is_dir() and manifest.is_file():
            found[child.name] = manifest
    return dict(sorted(found.items()))


def load_env_spec(env_yaml) -> dict:
    try:
        raw = yaml.safe_load(env_yaml.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise BackendSetupError(f"cannot read backend manifest {env_yaml}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise BackendSetupError(f"backend manifest must be a mapping: {env_yaml}")
    spec = dict(raw)
    version = spec.get("python_version")
    if not isinstance(version, str) or not _PYTHON_VERSION_RE.fullmatch(version):
        raise BackendSetupError(
            f"{env_yaml}: python_version must be an exact major.minor value (for example 3.10)"
        )
    for key in ("pip_packages", "extra_pip_args", "health_check_cmd"):
        value = spec.get(key, [])
        if value is not None and (
            not isinstance(value, list) or not all(isinstance(item, str) for item in value)
        ):
            raise BackendSetupError(f"{env_yaml}: {key} must be a list of strings")
    return spec


def _probe_python(command: Sequence[str], timeout: float = 10.0) -> str:
    try:
        result = subprocess.run(
            [
                *command,
                "-I",
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BackendSetupError(f"cannot execute Python {' '.join(command)!r}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise BackendSetupError(
            f"Python probe failed for {' '.join(command)!r}: {detail or f'rc={result.returncode}'}"
        )
    return result.stdout.strip()


def _candidate_commands(required_version: str, explicit: Optional[str]) -> Iterable[List[str]]:
    seen: set[tuple[str, ...]] = set()

    def emit(command: Sequence[str]):
        key = tuple(command)
        if key not in seen:
            seen.add(key)
            return list(command)
        return None

    if explicit:
        expanded = str(Path(explicit).expanduser())
        resolved = shutil.which(expanded) or expanded
        command = emit([resolved])
        if command:
            yield command
        return

    current = emit([sys.executable])
    if current:
        yield current
    for executable_name in (f"python{required_version}", f"python{required_version.replace('.', '')}"):
        executable = shutil.which(executable_name)
        if executable:
            command = emit([executable])
            if command:
                yield command
    if sys.platform == "win32":
        py_launcher = shutil.which("py")
        if py_launcher:
            command = emit([py_launcher, f"-{required_version}"])
            if command:
                yield command


def select_creator(required_version: str, explicit: Optional[str] = None) -> List[str]:
    failures: list[str] = []
    for command in _candidate_commands(required_version, explicit):
        try:
            actual = _probe_python(command)
        except BackendSetupError as exc:
            failures.append(str(exc))
            continue
        if actual == required_version:
            return command
        failures.append(f"{' '.join(command)} is Python {actual}, expected {required_version}")
    detail = f" Details: {'; '.join(failures)}" if failures else ""
    raise BackendSetupError(
        f"no Python {required_version} interpreter found; pass --python /absolute/path/to/python.{detail}"
    )


def _validate_existing_python(py: Path, required_version: str) -> None:
    if not py.is_file():
        raise BackendSetupError(f"environment Python is missing: {py}")
    actual = _probe_python([str(py)])
    if actual != required_version:
        raise BackendSetupError(
            f"environment {py} uses Python {actual}; expected {required_version}. Recreate it with --force."
        )


def create_venv(creator: Sequence[str], venv_path: Path, *, dry_run: bool) -> None:
    # The creator can be launched from an activated control environment.  Use
    # isolated mode so its PYTHONPATH/user site cannot shadow the stdlib venv
    # module or leak packages into the backend environment being created.
    command = [*creator, "-I", "-m", "venv", str(venv_path)]
    print(f"  - creating venv: {venv_path}")
    print(f"  - creator: {' '.join(creator)}")
    if dry_run:
        return
    venv_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)


def pip_install(
    py: Path,
    packages: Sequence[str],
    extra_args: Sequence[str],
    *,
    dry_run: bool,
) -> None:
    if not packages:
        return
    commands = (
        [str(py), "-I", "-m", "pip", "install", "--upgrade", "pip"],
        [str(py), "-I", "-m", "pip", "install", *extra_args, *packages],
    )
    print(f"  - installing: {' '.join(packages)}")
    if dry_run:
        for command in commands:
            print(f"    $ {' '.join(command)}")
        return
    for command in commands:
        subprocess.run(command, check=True)


def run_health_check(spec: EnvSpec, *, timeout: float, dry_run: bool) -> None:
    command = spec.resolved_health_check_cmd()
    if not command:
        if not dry_run and not spec.is_ready():
            raise BackendSetupError(f"environment Python is not runnable: {spec.python_executable}")
        return
    print(f"  - health check: {' '.join(command)}")
    if dry_run:
        return
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BackendSetupError(f"health check could not run: {exc}") from exc
    if result.stdout.strip():
        print(f"    {result.stdout.strip()}")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise BackendSetupError(
            f"health check failed rc={result.returncode}: {detail or 'no output'}"
        )


def setup_one(
    name: str,
    env_yaml,
    *,
    dry_run: bool,
    force: bool,
    python: Optional[str],
    health_timeout: float,
    install_into_external: bool,
) -> None:
    manifest = load_env_spec(env_yaml)
    required_version = manifest["python_version"]
    runtime = backend_env_spec(
        name,
        required_packages=manifest.get("pip_packages") or [],
        extra_pip_args=manifest.get("extra_pip_args") or [],
        cuda_constraint=manifest.get("cuda_constraint"),
        health_check_cmd=manifest.get("health_check_cmd"),
    )
    py = runtime.python_executable
    print(f"\n=== backend: {name} ===")
    print(f"  - Python requirement: {required_version}")
    print(f"  - Python executable: {py}")
    if runtime.cuda_constraint:
        print(
            f"  - CUDA compatibility target: {runtime.cuda_constraint} "
            "(informational; verify torch/vLLM against the host driver)"
        )

    if not runtime.managed:
        if force:
            raise BackendSetupError("--force cannot recreate an externally managed Python override")
        _validate_existing_python(py, required_version)
        if install_into_external:
            pip_install(
                py,
                runtime.required_packages,
                runtime.extra_pip_args,
                dry_run=dry_run,
            )
        else:
            print("  - external Python override: skipping package installation")
        run_health_check(runtime, timeout=health_timeout, dry_run=dry_run)
        return

    if runtime.venv_path.exists() and force:
        print(f"  - removing existing venv: {runtime.venv_path}")
        if not dry_run:
            shutil.rmtree(runtime.venv_path)
            if runtime.venv_path.exists():
                raise BackendSetupError(f"failed to remove environment: {runtime.venv_path}")

    if py.is_file() and not force:
        _validate_existing_python(py, required_version)
        print("  - existing environment has the required Python")
    else:
        if runtime.venv_path.exists() and not dry_run:
            raise BackendSetupError(
                f"environment exists but its Python is missing: {runtime.venv_path}; use --force"
            )
        creator = select_creator(required_version, explicit=python)
        create_venv(creator, runtime.venv_path, dry_run=dry_run)

    pip_install(
        py,
        runtime.required_packages,
        runtime.extra_pip_args,
        dry_run=dry_run,
    )
    run_health_check(runtime, timeout=health_timeout, dry_run=dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create isolated Python environments for DistillWheel backends."
    )
    parser.add_argument("backends", nargs="*", help="backend names; default: all")
    parser.add_argument("--python", help="creator Python executable (must match manifest version)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="recreate a managed environment")
    parser.add_argument("--health-timeout", type=float, default=120.0)
    parser.add_argument(
        "--install-into-external",
        action="store_true",
        help="allow pip installation into DISTILLWHEEL_<BACKEND>_PYTHON",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.health_timeout <= 0:
        print("error: --health-timeout must be > 0", file=sys.stderr)
        return 2
    manifests = discover_backends()
    if not manifests:
        print("error: no packaged backend env.yaml manifests found", file=sys.stderr)
        return 1
    targets = args.backends or list(manifests)
    unknown = [name for name in targets if name not in manifests]
    if unknown:
        print(f"error: unknown backend(s) {unknown}; known={list(manifests)}", file=sys.stderr)
        return 2

    failures: list[str] = []
    for name in targets:
        try:
            setup_one(
                name,
                manifests[name],
                dry_run=args.dry_run,
                force=args.force,
                python=args.python,
                health_timeout=args.health_timeout,
                install_into_external=args.install_into_external,
            )
        except (BackendSetupError, subprocess.CalledProcessError, OSError) as exc:
            failures.append(f"{name}: {exc}")
            print(f"error: {name}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BackendSetupError",
    "discover_backends",
    "load_env_spec",
    "main",
    "run_health_check",
    "select_creator",
    "setup_one",
]
