"""Create the isolated venvs declared by each backend's ``env.yaml``.

This is the *only* main-process tool that knows about backend-specific
package lists. Backends themselves never call pip — that's intentional:
all dependency installation happens here, before any training run.

Usage::

    python tools/setup_backend_envs.py              # all backends
    python tools/setup_backend_envs.py swift        # one backend
    python tools/setup_backend_envs.py --dry-run    # show actions only
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKENDS_DIR = REPO_ROOT / "distillwheel" / "backends"


def discover_backends() -> dict:
    """Return ``{name: env.yaml path}`` for every backend that has one."""
    out = {}
    for child in BACKENDS_DIR.iterdir():
        if not child.is_dir():
            continue
        env_yaml = child / "env.yaml"
        if env_yaml.exists():
            out[child.name] = env_yaml
    return out


def load_env_spec(env_yaml: Path) -> dict:
    with open(env_yaml, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def venv_python(venv_path: Path) -> Path:
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def create_venv(venv_path: Path, *, dry_run: bool) -> None:
    print(f"  - creating venv: {venv_path}")
    if dry_run:
        return
    venv_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "venv", str(venv_path)])


def pip_install(py: Path, packages: List[str], extra_args: List[str], *, dry_run: bool) -> None:
    if not packages:
        return
    cmd = [str(py), "-m", "pip", "install", "--upgrade", "pip"]
    print(f"  - upgrading pip")
    if not dry_run:
        subprocess.check_call(cmd)

    cmd = [str(py), "-m", "pip", "install", *extra_args, *packages]
    print(f"  - installing: {' '.join(packages)}")
    if not dry_run:
        subprocess.check_call(cmd)


def setup_one(name: str, env_yaml: Path, *, dry_run: bool, force: bool) -> None:
    spec = load_env_spec(env_yaml)
    venv_rel = Path(spec.get("venv_path", f".venvs/{name}"))
    venv_abs = (REPO_ROOT / venv_rel).resolve() if not venv_rel.is_absolute() else venv_rel

    py = venv_python(venv_abs)
    print(f"\n=== backend: {name} ({venv_abs}) ===")

    if py.exists() and not force:
        print(f"  - venv already exists; pass --force to recreate")
    else:
        if py.exists() and force:
            print(f"  - removing existing venv")
            if not dry_run:
                shutil.rmtree(venv_abs, ignore_errors=True)
        create_venv(venv_abs, dry_run=dry_run)

    pip_install(
        py,
        list(spec.get("pip_packages", [])),
        list(spec.get("extra_pip_args", [])),
        dry_run=dry_run,
    )

    hc = spec.get("health_check_cmd")
    if hc and not dry_run:
        # replace literal "python" with this venv's python
        hc_resolved = [str(py) if x == "python" else x for x in hc]
        print(f"  - health check: {' '.join(hc_resolved)}")
        try:
            subprocess.check_call(hc_resolved)
        except subprocess.CalledProcessError as e:
            print(f"  ! health check failed: {e}", file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Create isolated venvs for distillwheel backends.")
    parser.add_argument("backends", nargs="*", help="names to set up; default: all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="recreate venv if it exists")
    args = parser.parse_args(argv)

    all_backends = discover_backends()
    if not all_backends:
        print("no backends with env.yaml found.")
        return 1

    targets = args.backends or list(all_backends)
    for name in targets:
        if name not in all_backends:
            print(f"unknown backend: {name}. known={list(all_backends)}", file=sys.stderr)
            return 2
        setup_one(name, all_backends[name], dry_run=args.dry_run, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
