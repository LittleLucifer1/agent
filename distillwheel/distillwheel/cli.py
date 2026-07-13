"""``distillwheel`` CLI.

Examples::

    distillwheel run --recipe configs/qwen_sft_lora.yaml --data data/sft.jsonl
    distillwheel run --recipe configs/qwen_grpo.yaml --data data/prompts.jsonl

    distillwheel list-backends
    distillwheel show-route --recipe configs/qwen_grpo.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .core.errors import DistillWheelError
from .core.ir.recipe import Recipe
from .core.ir.sample import iter_samples_from_jsonl
from .core.registry import list_adapters, load_entry_points
from .core.router import resolve
from .pipeline.orchestrator import run_training
from .version import __version__


def _cmd_run(args: argparse.Namespace) -> int:
    recipe = Recipe.from_yaml(args.recipe)
    stream = iter_samples_from_jsonl(args.data)
    out = run_training(
        recipe,
        stream,
        skip_preflight=args.skip_preflight,
        heartbeat_timeout_s=args.heartbeat_timeout,
        overwrite_output=args.overwrite_output,
    )
    print(f"normalized model at: {out / 'final'}")
    return 0


def _cmd_list_backends(args: argparse.Namespace) -> int:
    load_entry_points()
    names = list_adapters()
    if not names:
        print("(no backends registered — install one or check entry_points)")
        return 1
    for n in names:
        print(n)
    return 0


def _cmd_show_route(args: argparse.Namespace) -> int:
    load_entry_points()
    recipe = Recipe.from_yaml(args.recipe)
    adapter = resolve(recipe)
    print(f"stage={recipe.stage} → backend={adapter.name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="distillwheel", description="Training-backend adapter framework.")
    p.add_argument("--version", action="version", version=f"distillwheel {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("run", help="run a training job from a Recipe + JSONL dataset")
    rp.add_argument("--recipe", required=True, type=Path)
    rp.add_argument("--data", required=True, type=Path)
    rp.add_argument("--skip-preflight", action="store_true")
    rp.add_argument("--heartbeat-timeout", type=_positive_float, default=None,
                    help="kill the subprocess if no stdout for N seconds")
    rp.add_argument(
        "--overwrite-output",
        action="store_true",
        help="replace managed files in an existing DistillWheel output directory",
    )
    rp.set_defaults(func=_cmd_run)

    lp = sub.add_parser("list-backends", help="list adapters currently registered")
    lp.set_defaults(func=_cmd_list_backends)

    sp = sub.add_parser("show-route", help="show which adapter would handle a recipe")
    sp.add_argument("--recipe", required=True, type=Path)
    sp.set_defaults(func=_cmd_show_route)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DistillWheelError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
