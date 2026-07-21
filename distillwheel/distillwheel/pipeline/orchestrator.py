"""Top-level training orchestrator.

The orchestrator is the only place that ties IR → adapter → launcher →
checkpoint/log normalizers together. It runs entirely in the main
process — the *training* itself happens inside the subprocess spawned by
the adapter's Launcher.
"""

from __future__ import annotations

import time
from contextlib import closing
from pathlib import Path
from typing import Optional

from ..core.errors import TrainingFailedError
from ..core.ir.recipe import Recipe
from ..core.ir.sample import SampleStream, validated_sample_stream
from ..core.registry import load_entry_points
from ..core.router import resolve
from .artifacts import build_output_layout
from .preflight import run_preflight


def run_training(
    recipe: Recipe,
    stream: SampleStream,
    *,
    skip_preflight: bool = False,
    heartbeat_timeout_s: Optional[float] = None,
    overwrite_output: bool = False,
) -> Path:
    """Run one training job end-to-end and return its managed run directory."""
    recipe.validate()
    load_entry_points()
    adapter = resolve(recipe)

    with build_output_layout(
        recipe.io.output_dir,
        overwrite=overwrite_output,
    ) as layout:
        # Save the original IR recipe at the run root for traceability.
        recipe.to_yaml(layout.recipe_yaml)

        # 1) Validate lazily and translate IR into native data.
        validated_stream = validated_sample_stream(stream, recipe.stage)
        data_path = adapter.prepare_data(validated_stream, recipe, layout.workdir)

        # 2) Recipe → native config (adapter also drops recipe.yaml in workdir)
        config_path = adapter.prepare_config(recipe, data_path, layout.workdir)

        # 3) Launcher
        launcher = adapter.build_launcher(config_path, recipe, layout.workdir)
        launcher.prepare_env()

        if not skip_preflight:
            run_preflight(adapter, recipe, data_path, layout.workdir)

        # 4) Run + persist logs + parse metrics
        parser = adapter.log_parser()
        parser.stage = recipe.stage
        t0 = time.monotonic()
        with open(layout.metrics_jsonl, "w", encoding="utf-8") as fp_metrics:
            # Explicitly close the launch generator if logging/parsing raises.
            # Its ``finally`` block is what terminates the entire subprocess
            # tree, and relying on garbage collection is not deterministic.
            with closing(launcher.launch(heartbeat_timeout_s=heartbeat_timeout_s)) as lines:
                for line in lines:
                    layout.append_raw_log(line)
                    metric = parser.parse_line(line)
                    if metric is not None:
                        fp_metrics.write(metric.to_json() + "\n")
                        fp_metrics.flush()
        duration = time.monotonic() - t0

        if launcher.returncode != 0:
            raise TrainingFailedError(
                returncode=launcher.returncode,
                tail=layout.tail_raw_log(n=80),
                message=(
                    f"backend={adapter.name} stage={recipe.stage} "
                    f"rc={launcher.returncode} duration={duration:.1f}s"
                ),
            )

        # 5) Normalize checkpoint into final/ + metadata.json
        native_dir = launcher.collect_artifacts()
        normalizer = adapter.checkpoint_normalizer()
        normalized = normalizer.normalize(
            native_dir=native_dir,
            output_dir=layout.root,
            recipe_yaml_path=layout.recipe_yaml,
        )
        layout.write_metadata(normalized)
        return layout.root
