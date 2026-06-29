"""Static check: `distillwheel.core.*` and each `backends/*/adapter.py`
must not import training frameworks (torch, swift, verl, ray, vllm, ...).

This guards the invariant that framework code only loads inside the
subprocess that the Launcher spawns.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Names that must not be imported from main-process code.
_FORBIDDEN = {
    "torch",
    "transformers",
    "swift",
    "verl",
    "ray",
    "vllm",
    "deepspeed",
    "peft",
    "trl",
}


def _imports_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                out.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module.split(".")[0])
    return out


def _gather_targets() -> list[Path]:
    paths: list[Path] = []
    paths.extend((REPO_ROOT / "distillwheel" / "core").rglob("*.py"))
    paths.extend((REPO_ROOT / "distillwheel" / "pipeline").rglob("*.py"))
    backends = REPO_ROOT / "distillwheel" / "backends"
    if backends.exists():
        for d in backends.iterdir():
            if d.is_dir():
                init = d / "__init__.py"
                adapter = d / "adapter.py"
                if init.exists():
                    paths.append(init)
                if adapter.exists():
                    paths.append(adapter)
    return paths


@pytest.mark.parametrize("path", _gather_targets(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_framework_imports_at_module_top(path):
    leaked = _imports_in(path) & _FORBIDDEN
    assert not leaked, (
        f"{path.relative_to(REPO_ROOT)} top-level imports forbidden modules: {sorted(leaked)}. "
        "Move that import into a function body (so it only runs in a subprocess)."
    )
