"""Output directory layout helpers.

The orchestrator creates this tree per run::

    {recipe.io.output_dir}/
    ├── workdir/               # framework-native intermediate files
    ├── raw_logs/              # full stdout from the subprocess
    ├── metrics.jsonl          # normalized metric stream
    ├── final/                 # normalized HF model (created by CheckpointNormalizer)
    ├── checkpoints/           # normalized intermediate checkpoints (optional)
    ├── training_recipe.yaml   # original IR recipe
    └── metadata.json          # NormalizedCheckpoint serialization
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from ..core.checkpoint import NormalizedCheckpoint


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

    # internal: append-only raw log file handle
    _raw_log_path: Path = None  # type: ignore[assignment]

    def append_raw_log(self, line: str) -> None:
        """Append a single stdout line to the raw log file."""
        with open(self._raw_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def tail_raw_log(self, n: int = 50) -> str:
        """Return the last ``n`` lines of the raw log (for error messages)."""
        try:
            with open(self._raw_log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return "".join(lines[-n:])
        except OSError:
            return ""

    def write_metadata(self, ck: NormalizedCheckpoint) -> None:
        with open(self.metadata_json, "w", encoding="utf-8") as f:
            json.dump(ck.to_dict(), f, indent=2, ensure_ascii=False)


def build_output_layout(output_dir: str | os.PathLike) -> OutputLayout:
    root = Path(output_dir).resolve()
    workdir = root / "workdir"
    raw_logs = root / "raw_logs"
    final_dir = root / "final"
    checkpoints_dir = root / "checkpoints"
    for p in (root, workdir, raw_logs, final_dir, checkpoints_dir):
        p.mkdir(parents=True, exist_ok=True)

    raw_log_path = raw_logs / "stdout.log"
    raw_log_path.touch(exist_ok=True)

    layout = OutputLayout(
        root=root,
        workdir=workdir,
        raw_logs_dir=raw_logs,
        metrics_jsonl=root / "metrics.jsonl",
        final_dir=final_dir,
        checkpoints_dir=checkpoints_dir,
        recipe_yaml=root / "training_recipe.yaml",
        metadata_json=root / "metadata.json",
    )
    layout._raw_log_path = raw_log_path
    return layout
