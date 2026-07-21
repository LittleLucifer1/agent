"""Checkpoint normalization — framework-native dir → HF safetensors layout.

归一化流程
==========

不同框架训练完后产出的 checkpoint 格式各异
本模块定义了统一的归一化接口 :class:`CheckpointNormalizer`，每个 backend
实现自己的子类，最终都产出相同的目录布局。

统一输出布局
============
    {output_dir}/
    ├── final/                                      # ← 最终可加载模型
    │   ├── config.json                             # HF 模型配置
    │   ├── tokenizer.json                          # tokenizer 文件
    │   ├── tokenizer_config.json
    │   ├── special_tokens_map.json
    │   ├── model.safetensors                       # 全参权重
    │   │   └── (或 adapter_model.safetensors)      # LoRA 权重
    │   ├── adapter_config.json                     # LoRA 时才有
    │   ├── training_recipe.yaml                    # 原始 IR recipe 副本
    │   └── metadata.json                           # NormalizedCheckpoint 序列化
    ├── checkpoints/                                # ← 中间 checkpoint (可选)
    │   ├── step_100/...
    │   └── step_200/...
    └── metadata.json                               # 根目录也保留一份 (由 orchestrator 写入)

metadata.json 示例
==================
{
    "final_dir": "/outputs/run_001/final",
    "is_lora": true,
    "step": 500,
    "base_model": "Qwen/Qwen2.5-7B",
    "framework": "swift",
    "extra": {"native_dir": "/outputs/run_001/workdir/swift_native"}
}
"""

from __future__ import annotations

import json
import os
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .errors import CheckpointError


@dataclass
class NormalizedCheckpoint:
    """归一化后的 checkpoint 元信息，序列化为 ``metadata.json``。"""
    final_dir: Path                              # 最终模型目录 (包含 safetensors + tokenizer)
    is_lora: bool                                # True = LoRA adapter, False = 全参模型
    step: Optional[int]                          # 对应的训练步数 (None = 未知)
    base_model: str                              # 基座模型名称/路径 (LoRA 加载时需要)
    framework: str                               # 产出该 checkpoint 的 backend 名称
    extra: dict = field(default_factory=dict)    # 附加元数据 (native_dir, recipe hash, git commit ...)

    def validate(self) -> None:
        if not isinstance(self.final_dir, Path):
            raise CheckpointError("checkpoint final_dir must be a Path")
        if not isinstance(self.is_lora, bool):
            raise CheckpointError("checkpoint is_lora must be a boolean")
        if self.step is not None and (
            not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0
        ):
            raise CheckpointError("checkpoint step must be a non-negative integer or null")
        if not isinstance(self.base_model, str):
            raise CheckpointError("checkpoint base_model must be a string")
        if not isinstance(self.framework, str) or not self.framework.strip():
            raise CheckpointError("checkpoint framework must be a non-empty string")
        if not isinstance(self.extra, dict):
            raise CheckpointError("checkpoint extra must be a mapping")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        d = asdict(self)
        d["final_dir"] = str(self.final_dir)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NormalizedCheckpoint":
        """从 metadata.json 反序列化。"""
        if not isinstance(d, Mapping):
            raise CheckpointError("checkpoint metadata must be a mapping")
        for required in ("final_dir", "is_lora"):
            if required not in d:
                raise CheckpointError(f"checkpoint metadata missing {required!r}")
        raw_final_dir = d["final_dir"]
        if not isinstance(raw_final_dir, (str, os.PathLike)) or not str(raw_final_dir).strip():
            raise CheckpointError("checkpoint final_dir must be a non-empty path")
        extra = d.get("extra", {})
        if not isinstance(extra, Mapping):
            raise CheckpointError("checkpoint extra must be a mapping")
        checkpoint = cls(
            final_dir=Path(raw_final_dir),
            is_lora=d["is_lora"],
            step=d.get("step"),
            base_model=d.get("base_model", ""),
            framework=d.get("framework", "unknown"),
            extra=dict(extra),
        )
        checkpoint.validate()
        return checkpoint

    @classmethod
    def from_json(cls, path: str | Path) -> "NormalizedCheckpoint":
        """从 metadata.json 文件加载。"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        except CheckpointError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"cannot read checkpoint metadata {path}: {exc}") from exc


class CheckpointNormalizer(ABC):
    """把框架原生 checkpoint 转成统一 HF 目录布局。

    每个 backend 实现子类:
    - :class:`SwiftCheckpointNormalizer` — 复制/重命名文件即可
    - :class:`VerlCheckpointNormalizer` — 调 model_merger 合并 FSDP 分片

    子类在 :meth:`normalize` 内部应依次调用:

    1. ``_ensure_final_dir()`` — 创建 ``output_dir/final/``
    2. (框架特定的权重复制/合并逻辑)
    3. ``_copy_recipe()`` — 把 recipe.yaml 复制到 final/
    4. ``_write_metadata()`` — 写入 final/metadata.json
    """

    framework: str = "unknown"

    @abstractmethod
    def normalize(
        self,
        native_dir: Path,
        output_dir: Path,
        recipe_yaml_path: Path,
    ) -> NormalizedCheckpoint:
        """执行归一化，返回元信息。

        Parameters
        ----------
        native_dir : Path
            框架原生输出目录 (来自 ``Launcher.collect_artifacts()``)
        output_dir : Path
            统一输出根目录 (来自 ``OutputLayout.root``)
        recipe_yaml_path : Path
            原始 IR recipe 文件路径 (复制到 final/ 保证可复现)
        """

    # ---------- helpers shared by subclasses ----------

    def _ensure_final_dir(self, output_dir: Path) -> Path:
        """创建并返回 ``output_dir/final/`` 目录。"""
        final = Path(output_dir) / "final"
        try:
            final.mkdir(parents=True, exist_ok=True)
            if any(final.iterdir()):
                raise CheckpointError(
                    f"refusing to normalize into non-empty checkpoint directory: {final}"
                )
        except CheckpointError:
            raise
        except OSError as exc:
            raise CheckpointError(f"cannot create normalized checkpoint directory {final}: {exc}") from exc
        return final

    def _copy_recipe(self, recipe_yaml_path: Path, final_dir: Path) -> None:
        """把 IR recipe 原文复制到 final/ 目录，保证模型可复现。"""
        import shutil

        src = Path(recipe_yaml_path)
        if not src.is_file():
            raise CheckpointError(f"training recipe does not exist or is not a file: {src}")
        try:
            shutil.copy2(src, final_dir / "training_recipe.yaml")
        except OSError as exc:
            raise CheckpointError(f"cannot copy training recipe {src}: {exc}") from exc

    def _write_metadata(self, ck: NormalizedCheckpoint) -> None:
        """把 NormalizedCheckpoint 序列化为 ``final/metadata.json``。"""
        meta = ck.to_dict()
        path = ck.final_dir / "metadata.json"
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False, allow_nan=False)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except (OSError, TypeError, ValueError) as exc:
            raise CheckpointError(f"cannot write checkpoint metadata {path}: {exc}") from exc
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
