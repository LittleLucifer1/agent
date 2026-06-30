"""swift checkpoint normalizer.

swift already writes HF-compatible safetensors with a tokenizer next to
it, so normalization mostly means copying / renaming a few files.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from ...core.checkpoint import CheckpointNormalizer, NormalizedCheckpoint
from ...core.errors import CheckpointError


# Files that we expect to find next to a final swift checkpoint
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
)
_MODEL_FILES_FULL = ("model.safetensors", "pytorch_model.bin")
_MODEL_FILES_LORA = ("adapter_model.safetensors", "adapter_model.bin")


class SwiftCheckpointNormalizer(CheckpointNormalizer):
    framework = "swift"

    def normalize(
        self,
        native_dir: Path,
        output_dir: Path,
        recipe_yaml_path: Path,
    ) -> NormalizedCheckpoint:
        native_dir = Path(native_dir)
        if not native_dir.exists():
            raise CheckpointError(f"swift native dir missing: {native_dir}")

        # swift v4 creates a versioned subdir (v0-YYYYMMDD-HHMMSS/) inside
        # the output_dir; v2/v3 write directly into output_dir.
        effective_root = self._find_effective_root(native_dir)

        latest_dir, step = self._latest_checkpoint(effective_root)
        if latest_dir is None:
            latest_dir, step = effective_root, None

        final_dir = self._ensure_final_dir(output_dir)
        is_lora = self._copy_into_final(latest_dir, final_dir)

        # Also surface the base model config when training was LoRA: HF
        # adapter loaders need the base config at the same path or via
        # adapter_config.json's "base_model_name_or_path".
        if is_lora:
            cfg = latest_dir / "config.json"
            if not cfg.exists():
                # leave a marker so users notice
                (final_dir / "BASE_MODEL_REQUIRED.txt").write_text(
                    "LoRA-only checkpoint. Base model required for inference."
                )

        self._copy_recipe(Path(recipe_yaml_path), final_dir)

        # mirror non-final checkpoints under output_dir/checkpoints/
        self._mirror_intermediate(effective_root, Path(output_dir) / "checkpoints")

        base_model = self._read_base_model(latest_dir) or ""
        ck = NormalizedCheckpoint(
            final_dir=final_dir,
            is_lora=is_lora,
            step=step,
            base_model=base_model,
            framework=self.framework,
            extra={"native_dir": str(native_dir)},
        )
        self._write_metadata(ck)
        return ck

    # -------- helpers --------

    def _find_effective_root(self, native_dir: Path) -> Path:
        """Find the actual model output root.

        swift v4 nests output inside a versioned subdir like ``v0-20260630-160730/``.
        Pick the latest one; fall back to native_dir itself for older swift.
        """
        versioned = []
        for child in native_dir.iterdir():
            if child.is_dir() and child.name.startswith("v") and "-" in child.name:
                versioned.append(child)
        if versioned:
            versioned.sort(key=lambda p: p.name)
            return versioned[-1]
        return native_dir

    def _latest_checkpoint(self, native_dir: Path) -> tuple[Optional[Path], Optional[int]]:
        best = None
        best_step = -1
        for child in native_dir.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if name.startswith("checkpoint-"):
                try:
                    step = int(name.split("-", 1)[1])
                except ValueError:
                    continue
                if step > best_step:
                    best_step = step
                    best = child
        if best is None:
            return None, None
        return best, best_step

    def _copy_into_final(self, src: Path, final: Path) -> bool:
        is_lora = any((src / m).exists() for m in _MODEL_FILES_LORA)
        model_files = _MODEL_FILES_LORA if is_lora else _MODEL_FILES_FULL

        # config + tokenizer
        for name in ("config.json", "generation_config.json", "adapter_config.json", *_TOKENIZER_FILES):
            f = src / name
            if f.exists():
                shutil.copy2(f, final / name)

        # weights (copy first matching file)
        for name in model_files:
            f = src / name
            if f.exists():
                shutil.copy2(f, final / name)
                break
        else:
            raise CheckpointError(
                f"no model weights found in {src} "
                f"(expected one of {model_files})"
            )

        # also copy any sharded safetensors index
        for f in src.glob("model-*-of-*.safetensors"):
            shutil.copy2(f, final / f.name)
        idx = src / "model.safetensors.index.json"
        if idx.exists():
            shutil.copy2(idx, final / "model.safetensors.index.json")

        return is_lora

    def _read_base_model(self, src: Path) -> Optional[str]:
        import json

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
            if child.is_dir() and child.name.startswith("checkpoint-"):
                target = dest / child.name
                if target.exists():
                    continue
                try:
                    shutil.copytree(child, target)
                except (OSError, shutil.Error):
                    # best-effort; final/ is the source of truth
                    pass
