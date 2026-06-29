"""Checkpoint normalization — framework-native dir → HF safetensors layout."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class NormalizedCheckpoint:
    final_dir: Path
    is_lora: bool
    step: Optional[int]
    base_model: str
    framework: str
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["final_dir"] = str(self.final_dir)
        return d


class CheckpointNormalizer(ABC):
    """Convert a framework-native checkpoint dir into the unified layout.

    Target layout (rooted at ``output_dir`` from the orchestrator)::

        final/
            config.json
            tokenizer.json / tokenizer_config.json / special_tokens_map.json
            model.safetensors                 (full param) OR
            adapter_model.safetensors         (LoRA)
            adapter_config.json               (LoRA only)
            training_recipe.yaml              (copy of the IR recipe)
            metadata.json                     (NormalizedCheckpoint JSON)
        checkpoints/
            step_{N}/...                      (intermediate, optional)
    """

    framework: str = "unknown"

    @abstractmethod
    def normalize(
        self,
        native_dir: Path,
        output_dir: Path,
        recipe_yaml_path: Path,
    ) -> NormalizedCheckpoint:
        ...

    # ---------- helpers shared by subclasses ----------

    def _ensure_final_dir(self, output_dir: Path) -> Path:
        final = Path(output_dir) / "final"
        final.mkdir(parents=True, exist_ok=True)
        return final

    def _copy_recipe(self, recipe_yaml_path: Path, final_dir: Path) -> None:
        import shutil

        src = Path(recipe_yaml_path)
        if src.exists():
            shutil.copy2(src, final_dir / "training_recipe.yaml")

    def _write_metadata(self, ck: NormalizedCheckpoint) -> None:
        meta = ck.to_dict()
        with open(ck.final_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
