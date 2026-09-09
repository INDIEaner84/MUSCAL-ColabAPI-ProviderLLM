"""Dataset loading and validation for the three HF-based tracks (text, MoE, VL).

The audio track has its own preprocessing path (``liquid_audio``); see
``notebooks/04_lfm_audio_ft.ipynb``.

Supported input shapes
----------------------
* **Conversational** (recommended): ``{"messages": [{"role": ..., "content": ...}, ...]}``
* **Vision**: same, but ``content`` is a list of typed parts and a parallel
  ``images`` column carries the decoded RGB images.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, load_dataset

VALID_ROLES = {"system", "user", "assistant", "tool"}
IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


class DatasetError(ValueError):
    """Raised when a dataset does not match the shape the trainer expects."""


def load_split(cfg) -> Dataset:
    """Load the training split according to ``cfg.data``."""
    ds = _load_raw(cfg.data, split=cfg.data.dataset_split)
    if cfg.data.max_samples:
        ds = ds.select(range(min(cfg.data.max_samples, len(ds))))
    return ds


def load_eval_split(cfg) -> Dataset | None:
    data = cfg.data
    if data.eval_file:
        ds = _load_file(data.eval_file)
    elif data.eval_split and data.dataset_name:
        ds = load_dataset(
            data.dataset_name,
            data.dataset_config,
            split=data.eval_split,
        )
    else:
        return None
    if data.eval_samples:
        ds = ds.select(range(min(data.eval_samples, len(ds))))
    return ds


def _load_raw(data, split: str) -> Dataset:
    if data.train_file:
        return _load_file(data.train_file)
    if data.dataset_name:
        return load_dataset(data.dataset_name, data.dataset_config, split=split)
    raise DatasetError("either `data.train_file` or `data.dataset_name` must be set")


def _load_file(path: str | Path) -> Dataset:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"dataset file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise DatasetError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
        return Dataset.from_list(rows)
    if suffix == ".json":
        return Dataset.from_list(json.loads(path.read_text(encoding="utf-8")))
    if suffix in {".csv", ".tsv"}:
        return load_dataset("csv", data_files=str(path), split="train")
    if suffix in {".parquet", ".arrow"}:
        return load_dataset("parquet", data_files=str(path), split="train")
    raise DatasetError(f"unsupported dataset file type: {suffix}")


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


def load_file(path: str | Path) -> Dataset:
    """Load a local dataset file (jsonl/json/csv/parquet). Public alias of the loader."""
    return _load_file(path)


def validate_conversational(ds: Dataset, *, require_assistant: bool = True) -> None:
    """Fail loudly on malformed rows instead of dying 20 minutes into training."""
    if "messages" not in ds.column_names:
        raise DatasetError(
            "dataset needs a `messages` column; found "
            f"{ds.column_names}. See docs/datasets.md for the expected shapes."
        )
    for i, row in enumerate(ds):
        messages = row["messages"]
        if not isinstance(messages, list) or not messages:
            raise DatasetError(f"row {i}: `messages` must be a non-empty list")
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg:
                raise DatasetError(f"row {i}: every message needs a `role`")
            if msg["role"] not in VALID_ROLES:
                raise DatasetError(
                    f"row {i}: unknown role {msg['role']!r} (allowed: {sorted(VALID_ROLES)})"
                )
            if "content" not in msg:
                raise DatasetError(f"row {i}: message from {msg['role']} has no `content`")
        if require_assistant and messages[-1]["role"] != "assistant":
            raise DatasetError(
                f"row {i}: last message must be from `assistant`, got {messages[-1]['role']!r}"
            )


def dataset_report(ds: Dataset) -> dict[str, Any]:
    """Cheap sanity stats: rows, roles, length distribution."""
    total = len(ds)
    roles: dict[str, int] = {}
    lengths: list[int] = []
    for row in ds:
        for msg in row["messages"]:
            roles[msg["role"]] = roles.get(msg["role"], 0) + 1
        lengths.append(len(json.dumps(row["messages"], ensure_ascii=False)))
    lengths.sort()
    return {
        "rows": total,
        "messages_by_role": roles,
        "chars_per_row_p50": lengths[total // 2] if total else 0,
        "chars_per_row_p95": lengths[min(total - 1, int(total * 0.95))] if total else 0,
        "chars_per_row_max": lengths[-1] if total else 0,
    }


# --------------------------------------------------------------------------- #
# vision track
# --------------------------------------------------------------------------- #


def build_vlm_conversations(
    ds: Dataset,
    *,
    image_root: str | Path | None = None,
    image_column: str = "image",
    question_column: str = "question",
    answer_column: str = "answer",
    system_prompt: str | None = None,
):
    """Turn a flat VLM dataset into a list of LFM2.5-VL conversations.

    Expects rows with an image (path or PIL image), a question and an answer.
    Returns a plain Python list -- TRL's SFTTrainer is given a custom collator
    and ``skip_prepare_dataset=True`` for this track.
    """
    from PIL import Image

    root = Path(image_root) if image_root else None
    conversations = []
    for i, row in enumerate(ds):
        image = row[image_column]
        if isinstance(image, str):
            path = Path(image)
            if not path.is_absolute() and root is not None:
                path = root / path
            if not path.exists():
                raise DatasetError(f"row {i}: image not found: {path}")
            image = Image.open(path).convert("RGB")
        elif hasattr(image, "convert"):
            image = image.convert("RGB")
        else:
            raise DatasetError(f"row {i}: unsupported image value of type {type(image)!r}")

        turns = []
        if system_prompt:
            turns.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        turns.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": str(row[question_column])},
                ],
            }
        )
        turns.append(
            {"role": "assistant", "content": [{"type": "text", "text": str(row[answer_column])}]}
        )
        conversations.append(turns)
    return conversations


def find_image_files(root: str | Path) -> Iterable[Path]:
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in IMAGE_TYPES:
            yield path
