"""Normalize ms-swift checkpoints into DistillWheel's HF-style layout."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

from ...core.checkpoint import CheckpointNormalizer, NormalizedCheckpoint
from ...core.errors import CheckpointError


_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
    "chat_template.jinja",
)
_PROCESSOR_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "image_processor_config.json",
    "feature_extractor_config.json",
)
_AUXILIARY_FILES = (
    "config.json",
    "generation_config.json",
    "adapter_config.json",
    *_TOKENIZER_FILES,
    *_PROCESSOR_FILES,
    # Native arguments are useful provenance and can also help ms-swift tools
    # discover the original base model/template during later inference.
    "args.json",
)
_MODEL_FILES_FULL = ("model.safetensors", "pytorch_model.bin")
_MODEL_FILES_LORA = ("adapter_model.safetensors", "adapter_model.bin")
_MODEL_INDEX_FILES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


class SwiftCheckpointNormalizer(CheckpointNormalizer):
    framework = "swift"

    def normalize(
        self,
        native_dir: Path,
        output_dir: Path,
        recipe_yaml_path: Path,
    ) -> NormalizedCheckpoint:
        native_dir = Path(native_dir).expanduser().resolve()
        if not native_dir.exists():
            raise CheckpointError(f"swift native dir missing: {native_dir}")

        # add_version=false gives direct checkpoint-* directories.  Recursive
        # discovery keeps runs made with older defaults (vN-timestamp), and
        # last/best pointers, normalizable as well.
        latest_dir, step = self._latest_checkpoint(native_dir)
        if latest_dir is None:
            latest_dir = self._fallback_weight_dir(native_dir)
        if latest_dir is None:
            raise CheckpointError(f"no model weights found under {native_dir}")

        final_dir = self._ensure_final_dir(output_dir)
        is_lora = self._copy_into_final(latest_dir, final_dir)
        self._copy_ancestor_assets(latest_dir, native_dir, final_dir)

        if is_lora and not (latest_dir / "config.json").exists():
            (final_dir / "BASE_MODEL_REQUIRED.txt").write_text(
                "LoRA-only checkpoint. Base model required for inference.",
                encoding="utf-8",
            )

        self._copy_recipe(Path(recipe_yaml_path), final_dir)
        self._mirror_intermediate(native_dir, Path(output_dir) / "checkpoints")

        ck = NormalizedCheckpoint(
            final_dir=final_dir,
            is_lora=is_lora,
            step=step,
            base_model=self._read_base_model(latest_dir) or "",
            framework=self.framework,
            extra={"native_dir": str(native_dir)},
        )
        self._write_metadata(ck)
        return ck

    # -------- discovery --------

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
        candidates: list[tuple[int, str, Path]] = []
        for child in native_dir.rglob("checkpoint-*"):
            if not child.is_dir():
                continue
            match = _CHECKPOINT_RE.fullmatch(child.name)
            if match and self._has_model_weights(child):
                candidates.append((int(match.group(1)), str(child), child))
        if candidates:
            step, _, path = max(candidates, key=lambda item: (item[0], item[1]))
            return path.resolve(), step

        # Symlinked marker directories are not followed by rglob on every
        # supported Python/platform combination, so resolve them explicitly.
        for marker_name in ("last", "best"):
            markers = []
            direct = native_dir / marker_name
            if direct.exists() or direct.is_symlink():
                markers.append(direct)
            markers.extend(native_dir.rglob(marker_name))
            seen: set[Path] = set()
            for marker in markers:
                target = self._resolve_marker(marker)
                if target is None or target in seen:
                    continue
                seen.add(target)
                if self._has_model_weights(target):
                    return target, self._step_from_name(target.name)
                nested = self._highest_checkpoint_below(target)
                if nested is not None:
                    return nested, self._step_from_name(nested.name)
        return None, None

    def _highest_checkpoint_below(self, root: Path) -> Optional[Path]:
        candidates: list[tuple[int, str, Path]] = []
        if not root.is_dir():
            return None
        for child in root.rglob("checkpoint-*"):
            match = _CHECKPOINT_RE.fullmatch(child.name)
            if child.is_dir() and match and self._has_model_weights(child):
                candidates.append((int(match.group(1)), str(child), child))
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item[0], item[1]))[2].resolve()

    def _resolve_marker(self, marker: Path) -> Optional[Path]:
        try:
            if marker.is_dir() or marker.is_symlink():
                target = marker.resolve()
                return target if target.is_dir() else None
            if marker.is_file():
                raw = marker.read_text(encoding="utf-8").strip()
                if not raw:
                    return None
                target = Path(raw).expanduser()
                if not target.is_absolute():
                    target = marker.parent / target
                target = target.resolve()
                return target if target.is_dir() else None
        except OSError:
            return None
        return None

    def _fallback_weight_dir(self, native_dir: Path) -> Optional[Path]:
        if self._has_model_weights(native_dir):
            return native_dir
        candidates = [
            child
            for child in native_dir.rglob("*")
            if child.is_dir() and self._has_model_weights(child)
        ]
        if not candidates:
            return None

        def sort_key(path: Path) -> tuple[int, str]:
            try:
                mtime = path.stat().st_mtime_ns
            except OSError:
                mtime = -1
            return mtime, str(path)

        return max(candidates, key=sort_key).resolve()

    @staticmethod
    def _step_from_name(name: str) -> Optional[int]:
        match = _CHECKPOINT_RE.fullmatch(name)
        return int(match.group(1)) if match else None

    @staticmethod
    def _has_model_weights(path: Path) -> bool:
        names = (*_MODEL_FILES_LORA, *_MODEL_FILES_FULL, *_MODEL_INDEX_FILES)
        return any((path / name).is_file() for name in names)

    # -------- copying --------

    def _copy_into_final(self, src: Path, final: Path) -> bool:
        is_lora = any((src / name).is_file() for name in _MODEL_FILES_LORA)

        # Validate/select weights before copying auxiliary files so a broken
        # shard index cannot look like a partially successful normalization.
        if is_lora:
            selected = next(
                (src / name for name in _MODEL_FILES_LORA if (src / name).is_file()),
                None,
            )
            assert selected is not None
            weight_plan = ("single", selected)
        else:
            selected = next(
                (src / name for name in _MODEL_FILES_FULL if (src / name).is_file()),
                None,
            )
            if selected is not None:
                weight_plan = ("single", selected)
            else:
                index = next(
                    (src / name for name in _MODEL_INDEX_FILES if (src / name).is_file()),
                    None,
                )
                if index is None:
                    raise CheckpointError(
                        f"no model weights found in {src} (expected one of "
                        f"{(*_MODEL_FILES_LORA, *_MODEL_FILES_FULL, *_MODEL_INDEX_FILES)})"
                    )
                shards = self._validate_shard_index(src, index)
                weight_plan = ("sharded", (index, shards))

        for name in _AUXILIARY_FILES:
            source = src / name
            if source.is_file():
                shutil.copy2(source, final / name)

        if weight_plan[0] == "single":
            selected = weight_plan[1]
            assert isinstance(selected, Path)
            shutil.copy2(selected, final / selected.name)
        else:
            index, shards = weight_plan[1]
            assert isinstance(index, Path)
            for relative, source in shards:
                target = final / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            shutil.copy2(index, final / index.name)

        return is_lora

    def _copy_ancestor_assets(self, src: Path, native_dir: Path, final: Path) -> None:
        """Fill assets saved at a version/output root rather than checkpoint."""
        try:
            src.relative_to(native_dir)
        except ValueError:
            return

        current = src.parent
        while current == native_dir or native_dir in current.parents:
            for name in _AUXILIARY_FILES:
                source = current / name
                target = final / name
                if source.is_file() and not target.exists():
                    shutil.copy2(source, target)
            if current == native_dir:
                break
            current = current.parent

    def _validate_shard_index(
        self,
        src: Path,
        index_path: Path,
    ) -> list[tuple[Path, Path]]:
        try:
            with open(index_path, "r", encoding="utf-8") as handle:
                index_data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"invalid model shard index {index_path}: {exc}") from exc

        weight_map = index_data.get("weight_map") if isinstance(index_data, Mapping) else None
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise CheckpointError(
                f"model shard index {index_path} has no non-empty weight_map"
            )

        src_root = src.resolve()
        shards: list[tuple[Path, Path]] = []
        seen: set[Path] = set()
        for shard_name in weight_map.values():
            if not isinstance(shard_name, str) or not shard_name:
                raise CheckpointError(
                    f"model shard index {index_path} contains an invalid shard name"
                )
            relative = Path(shard_name)
            if relative.is_absolute() or ".." in relative.parts:
                raise CheckpointError(
                    f"model shard index {index_path} references unsafe path {shard_name!r}"
                )
            source = (src / relative).resolve()
            try:
                source.relative_to(src_root)
            except ValueError as exc:
                raise CheckpointError(
                    f"model shard index {index_path} escapes checkpoint: {shard_name!r}"
                ) from exc
            if not source.is_file():
                raise CheckpointError(
                    f"model shard index {index_path} references missing shard "
                    f"{shard_name!r}"
                )
            if relative not in seen:
                seen.add(relative)
                shards.append((relative, source))
        return shards

    def _read_base_model(self, src: Path) -> Optional[str]:
        for name in ("adapter_config.json", "config.json"):
            path = src / name
            if not path.is_file():
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, Mapping):
                continue
            for key in (
                "base_model_name_or_path",
                "_name_or_path",
                "model_name_or_path",
            ):
                if data.get(key):
                    return str(data[key])
        return None

    def _mirror_intermediate(self, native_dir: Path, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        for child in native_dir.rglob("checkpoint-*"):
            if not child.is_dir() or _CHECKPOINT_RE.fullmatch(child.name) is None:
                continue
            try:
                relative = child.relative_to(native_dir)
            except ValueError:
                continue
            target = dest / relative
            if target.exists():
                continue
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(child, target)
            except (OSError, shutil.Error):
                # Intermediate checkpoints are best-effort; final/ is the
                # source of truth and has already been validated above.
                pass
