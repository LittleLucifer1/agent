"""verl checkpoint normalizer.

verl saves FSDP-sharded model state under ``checkpoints/global_step_N/``.
To get an HF-loadable artifact we invoke verl's own model_merger script
(``python -m verl.scripts.model_merger`` or ``verl.utils.checkpoint``)
**inside the verl venv** so we don't import verl from the main process.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from ...core.checkpoint import CheckpointNormalizer, NormalizedCheckpoint
from ...core.envspec import EnvSpec
from ...core.errors import CheckpointError


class VerlCheckpointNormalizer(CheckpointNormalizer):
    framework = "verl"

    def __init__(self, env_spec: Optional[EnvSpec] = None):
        # Allow injection so tests can swap in a venv that doesn't really
        # have verl installed.
        self._env_spec = env_spec

    def normalize(
        self,
        native_dir: Path,
        output_dir: Path,
        recipe_yaml_path: Path,
    ) -> NormalizedCheckpoint:
        native_dir = Path(native_dir)
        if not native_dir.exists():
            raise CheckpointError(f"verl native dir missing: {native_dir}")

        latest, step = self._latest_step(native_dir)
        if latest is None:
            raise CheckpointError(f"no global_step_* directory in {native_dir}")

        final_dir = self._ensure_final_dir(output_dir)

        merged = self._merge_with_verl_script(latest, final_dir)
        if not merged:
            # Fall back to copying anything HF-ish that's already in place
            # (e.g. when verl was run with auto-merge enabled).
            self._copy_hf_outputs(latest, final_dir)

        is_lora = (final_dir / "adapter_model.safetensors").exists()
        self._copy_tokenizer(latest, final_dir)
        self._copy_recipe(Path(recipe_yaml_path), final_dir)

        # mirror intermediate global_step_* under output_dir/checkpoints
        self._mirror_intermediate(native_dir, Path(output_dir) / "checkpoints")

        ck = NormalizedCheckpoint(
            final_dir=final_dir,
            is_lora=is_lora,
            step=step,
            base_model=self._read_base_model(latest) or "",
            framework=self.framework,
            extra={"native_dir": str(native_dir)},
        )
        self._write_metadata(ck)
        return ck

    # ---------- internals ----------

    def _latest_step(self, native_dir: Path) -> Tuple[Optional[Path], Optional[int]]:
        best = None
        best_step = -1
        for child in native_dir.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if name.startswith("global_step_") or name.startswith("checkpoint-"):
                try:
                    step = int(name.split("_")[-1].split("-")[-1])
                except ValueError:
                    continue
                if step > best_step:
                    best_step = step
                    best = child
        if best is None:
            return None, None
        return best, best_step

    def _merge_with_verl_script(self, src: Path, final_dir: Path) -> bool:
        """Run ``verl.scripts.model_merger`` in the verl venv. Returns True on success."""
        if self._env_spec is None or not self._env_spec.is_ready():
            return False
        py = str(self._env_spec.python_executable)
        cmd = [
            py, "-m", "verl.scripts.model_merger",
            "--local_dir", str(src),
            "--target_dir", str(final_dir),
            "--hf_upload_target", "",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise CheckpointError(f"verl model_merger failed to launch: {e}") from e
        if proc.returncode != 0:
            raise CheckpointError(
                f"verl model_merger rc={proc.returncode}:\n{proc.stderr[-2000:]}"
            )
        return True

    def _copy_hf_outputs(self, src: Path, final_dir: Path) -> None:
        """Backup path: copy any HF-format files that are already there."""
        candidates = [
            "config.json", "generation_config.json", "adapter_config.json",
            "model.safetensors", "pytorch_model.bin",
            "adapter_model.safetensors", "adapter_model.bin",
        ]
        copied_weights = False
        for name in candidates:
            f = src / name
            if f.exists():
                shutil.copy2(f, final_dir / name)
                if name.startswith(("model.", "adapter_model.", "pytorch_model.")):
                    copied_weights = True
        for f in src.glob("model-*-of-*.safetensors"):
            shutil.copy2(f, final_dir / f.name)
            copied_weights = True
        idx = src / "model.safetensors.index.json"
        if idx.exists():
            shutil.copy2(idx, final_dir / idx.name)
        if not copied_weights:
            raise CheckpointError(
                f"could not produce HF weights for {src}: verl model_merger "
                "not available and no pre-merged files in source."
            )

    def _copy_tokenizer(self, src: Path, final_dir: Path) -> None:
        for name in ("tokenizer.json", "tokenizer_config.json",
                     "special_tokens_map.json", "vocab.json", "merges.txt",
                     "tokenizer.model"):
            f = src / name
            if f.exists():
                shutil.copy2(f, final_dir / name)
            else:
                # verl often keeps the tokenizer in the model dir's parent
                parent_f = src.parent / name
                if parent_f.exists():
                    shutil.copy2(parent_f, final_dir / name)

    def _read_base_model(self, src: Path) -> Optional[str]:
        for name in ("adapter_config.json", "config.json"):
            f = src / name
            if not f.exists():
                continue
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            for k in ("base_model_name_or_path", "_name_or_path", "model_name_or_path"):
                if data.get(k):
                    return data[k]
        return None

    def _mirror_intermediate(self, native_dir: Path, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        for child in native_dir.iterdir():
            if child.is_dir() and (child.name.startswith("global_step_") or child.name.startswith("checkpoint-")):
                target = dest / child.name
                if target.exists():
                    continue
                try:
                    shutil.copytree(child, target)
                except (OSError, shutil.Error):
                    pass
