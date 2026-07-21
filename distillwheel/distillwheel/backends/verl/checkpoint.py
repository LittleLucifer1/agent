"""Normalize VERL 0.8 checkpoints into Hugging Face layout."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional, Tuple

from ...core.checkpoint import CheckpointNormalizer, NormalizedCheckpoint
from ...core.envspec import EnvSpec
from ...core.errors import CheckpointError
from ...core.launcher import filter_env

_TOKENIZER_NAMES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt",
    "tokenizer.model", "spiece.model", "chat_template.jinja",
    "preprocessor_config.json", "processor_config.json",
    "video_preprocessor_config.json",
)
_CONFIG_NAMES = (
    "config.json", "generation_config.json", "adapter_config.json",
    "model.safetensors.index.json", "pytorch_model.bin.index.json",
    "adapter_model.safetensors.index.json", "adapter_model.bin.index.json",
)

_FULL_EXACT = ("model.safetensors", "pytorch_model.bin")
_LORA_EXACT = ("adapter_model.safetensors", "adapter_model.bin")
_SHARDED_LAYOUTS = {
    "full": (
        ("model-*-of-*.safetensors", "model.safetensors.index.json"),
        ("pytorch_model-*-of-*.bin", "pytorch_model.bin.index.json"),
    ),
    "lora": (
        ("adapter_model-*-of-*.safetensors", "adapter_model.safetensors.index.json"),
        ("adapter_model-*-of-*.bin", "adapter_model.bin.index.json"),
    ),
}
_DISTRIBUTED_ENV_KEYS = (
    "NPROC_PER_NODE", "NNODES", "NODE_RANK", "RANK", "WORLD_SIZE",
    "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
)


class VerlCheckpointNormalizer(CheckpointNormalizer):
    framework = "verl"

    def __init__(self, env_spec: Optional[EnvSpec] = None):
        self._env_spec = env_spec

    def normalize(
        self,
        native_dir: Path,
        output_dir: Path,
        recipe_yaml_path: Path,
    ) -> NormalizedCheckpoint:
        native_dir = Path(native_dir).expanduser().resolve()
        if not native_dir.is_dir():
            raise CheckpointError(f"VERL native checkpoint directory is missing: {native_dir}")

        latest, step = self._latest_step(native_dir)
        if latest is None:
            raise CheckpointError(f"no global_step_* VERL checkpoint found in {native_dir}")
        actor_dir = latest / "actor"
        if not actor_dir.is_dir():
            raise CheckpointError(f"VERL actor checkpoint is missing: {actor_dir}")

        final_dir = self._ensure_final_dir(Path(output_dir).expanduser().resolve())
        premerged = self._find_premerged(actor_dir)
        if premerged is not None:
            self._copy_hf_outputs(premerged, final_dir)
        else:
            self._merge_with_verl_script(actor_dir, final_dir)
            if not self._has_hf_weights(final_dir):
                raise CheckpointError(
                    f"VERL model merger completed but produced no HF weights in {final_dir}"
                )

        recipe_path = Path(recipe_yaml_path).expanduser().resolve()
        recipe_base_model = self._base_model_from_recipe(recipe_path)
        sources = [
            final_dir,
            actor_dir / "huggingface",
            actor_dir,
            latest / "huggingface",
            latest,
        ]
        if recipe_base_model:
            local_base = Path(recipe_base_model).expanduser()
            if local_base.is_dir():
                sources.append(local_base.resolve())
        self._copy_auxiliary_configs(sources, final_dir)
        self._copy_tokenizer(sources, final_dir)
        self._copy_recipe(recipe_path, final_dir)

        # A filename glob alone is not enough to call an artifact normalized:
        # incomplete shards and adapter weights without PEFT metadata are not
        # loadable by Transformers/PEFT.  Validate the final copied result,
        # after config files have had a chance to come from actor/huggingface.
        is_lora = self._validate_final_checkpoint(final_dir)

        self._mirror_intermediate(native_dir, Path(output_dir).expanduser().resolve() / "checkpoints")
        base_model = recipe_base_model or self._read_base_model(sources) or ""
        if is_lora:
            requirement = "LoRA-only checkpoint. Base model required for inference."
            if base_model:
                requirement += f"\nBase model: {base_model}"
            (final_dir / "BASE_MODEL_REQUIRED.txt").write_text(
                requirement + "\n",
                encoding="utf-8",
            )
        checkpoint = NormalizedCheckpoint(
            final_dir=final_dir,
            is_lora=is_lora,
            step=step,
            base_model=base_model,
            framework=self.framework,
            extra={"native_dir": str(native_dir), "actor_dir": str(actor_dir)},
        )
        self._write_metadata(checkpoint)
        return checkpoint

    def _latest_step(self, native_dir: Path) -> Tuple[Optional[Path], Optional[int]]:
        best: Optional[Path] = None
        best_step = -1
        for child in native_dir.iterdir():
            if not child.is_dir():
                continue
            match = None
            if child.name.startswith("global_step_"):
                match = child.name.removeprefix("global_step_")
            elif child.name.startswith("checkpoint-"):
                match = child.name.removeprefix("checkpoint-")
            if match is None:
                continue
            try:
                step = int(match)
            except ValueError:
                continue
            if step > best_step:
                best, best_step = child, step
        return (best, best_step) if best is not None else (None, None)

    @staticmethod
    def _has_hf_weights(directory: Path) -> bool:
        if not directory.is_dir():
            return False
        if any((directory / name).is_file() for name in _FULL_EXACT + _LORA_EXACT):
            return True
        patterns = tuple(
            pattern
            for layouts in _SHARDED_LAYOUTS.values()
            for pattern, _ in layouts
        )
        return any(any(directory.glob(pattern)) for pattern in patterns)

    @staticmethod
    def _weight_kind(directory: Path) -> Optional[str]:
        """Classify deployable weight filenames, preferring a full model."""
        if any((directory / name).is_file() for name in _FULL_EXACT):
            return "full"
        if any(any(directory.glob(pattern)) for pattern, _ in _SHARDED_LAYOUTS["full"]):
            return "full"
        if any((directory / name).is_file() for name in _LORA_EXACT):
            return "lora"
        if any(any(directory.glob(pattern)) for pattern, _ in _SHARDED_LAYOUTS["lora"]):
            return "lora"
        return None

    @classmethod
    def _validate_final_checkpoint(cls, final_dir: Path) -> bool:
        """Validate a loadable HF/PEFT weight layout and return ``is_lora``."""
        kind = cls._weight_kind(final_dir)
        if kind is None:
            raise CheckpointError(f"no HF weights found in normalized VERL output {final_dir}")

        exact_names = _FULL_EXACT if kind == "full" else _LORA_EXACT
        has_exact = any((final_dir / name).is_file() for name in exact_names)
        if not has_exact:
            matching_layout = None
            for pattern, index_name in _SHARDED_LAYOUTS[kind]:
                if any(final_dir.glob(pattern)):
                    matching_layout = (pattern, index_name)
                    break
            assert matching_layout is not None  # guaranteed by _weight_kind
            _, index_name = matching_layout
            cls._validate_shard_index(final_dir, index_name)

        required_config = "adapter_config.json" if kind == "lora" else "config.json"
        if not (final_dir / required_config).is_file():
            raise CheckpointError(
                f"normalized VERL {kind} checkpoint is missing required {required_config}"
            )
        return kind == "lora"

    @staticmethod
    def _validate_shard_index(directory: Path, index_name: str) -> None:
        index_path = directory / index_name
        if not index_path.is_file():
            raise CheckpointError(
                f"sharded VERL checkpoint is missing required index {index_name}"
            )
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"invalid VERL shard index {index_path}: {exc}") from exc
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise CheckpointError(f"VERL shard index has no non-empty weight_map: {index_path}")
        shard_names = set(weight_map.values())
        if not shard_names or any(not isinstance(name, str) or not name for name in shard_names):
            raise CheckpointError(f"VERL shard index contains invalid shard names: {index_path}")
        for name in sorted(shard_names):
            if Path(name).name != name:
                raise CheckpointError(
                    f"VERL shard index must reference files in its own directory: {name!r}"
                )
            if not (directory / name).is_file():
                raise CheckpointError(
                    f"VERL shard index references a missing weight file: {name}"
                )

    def _find_premerged(self, actor_dir: Path) -> Optional[Path]:
        for candidate in (actor_dir / "huggingface", actor_dir):
            if self._has_hf_weights(candidate):
                return candidate
        return None

    def _merge_with_verl_script(self, actor_dir: Path, final_dir: Path) -> None:
        if self._env_spec is None or not self._env_spec.is_ready():
            expected = self._env_spec.python_executable if self._env_spec else "<unset>"
            raise CheckpointError(
                "VERL checkpoint is sharded and requires the VERL 0.8 model merger; "
                f"backend Python is not ready at {expected}"
            )
        python = self._env_spec.python_executable.expanduser().resolve()
        command = [
            str(python), "-I", "-m", "verl.model_merger", "merge",
            "--backend", "fsdp",
            "--local_dir", str(actor_dir.resolve()),
            "--target_dir", str(final_dir.resolve()),
        ]
        merger_base_env = dict(os.environ)
        for key in _DISTRIBUTED_ENV_KEYS:
            merger_base_env.pop(key, None)
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=1800,
                check=False,
                env=filter_env(
                    base_env=merger_base_env,
                    extra={"PYTHONUNBUFFERED": "1"},
                ),
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _process_text(exc.stdout)
            stderr = _process_text(exc.stderr)
            raise CheckpointError(
                "VERL model merger timed out after 1800s\n"
                f"stdout:\n{stdout[-4000:]}\nstderr:\n{stderr[-4000:]}"
            ) from exc
        except (OSError, ValueError) as exc:
            raise CheckpointError(f"failed to launch VERL model merger: {exc}") from exc
        if process.returncode != 0:
            raise CheckpointError(
                f"VERL model merger exited with code {process.returncode}\n"
                f"stdout:\n{process.stdout[-4000:]}\n"
                f"stderr:\n{process.stderr[-4000:]}"
            )

    def _copy_hf_outputs(self, source: Path, final_dir: Path) -> None:
        if not self._has_hf_weights(source):
            raise CheckpointError(f"no pre-merged HF weights found in {source}")
        names = set(_CONFIG_NAMES) | set(_TOKENIZER_NAMES)
        patterns = (
            "model*.safetensors", "pytorch_model*.bin",
            "adapter_model*.safetensors", "adapter_model*.bin",
        )
        files = {source / name for name in names if (source / name).is_file()}
        for pattern in patterns:
            files.update(path for path in source.glob(pattern) if path.is_file())
        for file in sorted(files):
            shutil.copy2(file, final_dir / file.name)

    @staticmethod
    def _copy_auxiliary_configs(sources: Iterable[Path], final_dir: Path) -> None:
        for name in ("config.json", "generation_config.json", "adapter_config.json"):
            destination = final_dir / name
            if destination.is_file():
                continue
            for source in sources:
                candidate = Path(source) / name
                if candidate.is_file():
                    shutil.copy2(candidate, destination)
                    break

    @staticmethod
    def _copy_tokenizer(sources: Iterable[Path], final_dir: Path) -> None:
        for name in _TOKENIZER_NAMES:
            destination = final_dir / name
            if destination.is_file():
                continue
            for source in sources:
                candidate = Path(source) / name
                if candidate.is_file():
                    shutil.copy2(candidate, destination)
                    break

    @staticmethod
    def _read_base_model(sources: Iterable[Path]) -> Optional[str]:
        for source in sources:
            for name in ("adapter_config.json", "config.json"):
                file = Path(source) / name
                if not file.is_file():
                    continue
                try:
                    data = json.loads(file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                for key in ("base_model_name_or_path", "_name_or_path", "model_name_or_path"):
                    value = data.get(key)
                    if isinstance(value, str) and value:
                        return value
        return None

    @staticmethod
    def _base_model_from_recipe(recipe_path: Path) -> Optional[str]:
        if not recipe_path.is_file():
            return None
        try:
            import yaml

            data = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError, TypeError):
            return None
        if isinstance(data, dict) and isinstance(data.get("base_model"), str):
            return data["base_model"]
        return None

    @staticmethod
    def _mirror_intermediate(native_dir: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        for child in native_dir.iterdir():
            if not child.is_dir() or not (
                child.name.startswith("global_step_") or child.name.startswith("checkpoint-")
            ):
                continue
            target = destination / child.name
            if target.exists():
                continue
            try:
                shutil.copytree(child, target)
            except (OSError, shutil.Error):
                # Mirroring is best-effort; the canonical final artifact above
                # has already been validated.
                pass


def _process_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
