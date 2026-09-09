"""Configuration objects for MUSCAL LFM fine-tuning runs.

Everything a run needs lives in one YAML file (see ``configs/``). The dataclasses
here are intentionally boring: they map 1:1 onto the YAML so that a config file is
the single source of truth for a run and can be diffed, copied and shared.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


# Default LoRA targets that work across the LFM2 / LFM2.5 family.
#
# LFM2.x is a hybrid backbone: gated short convolutions + grouped query attention.
# The "gate_proj/up_proj/down_proj" names cover the dense MLP blocks *and*, by
# suffix match, the expert MLPs inside MoE layers (LFM2.5-8B-A1B), so the same
# target list works for dense and sparse checkpoints.
DEFAULT_TARGET_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Vision track additionally touches the projector / connector linear layers.
VLM_EXTRA_TARGET_MODULES: list[str] = ["fc1", "fc2", "linear"]


@dataclass
class ModelConf:
    """Which checkpoint to load and how to load it."""

    id: str = "LiquidAI/LFM2.5-1.2B-Instruct"
    track: str = "text"  # text | moe | vl | audio
    attn_implementation: str = "sdpa"  # sdpa | eager | flash_attention_2
    trust_remote_code: bool = True


@dataclass
class QuantConf:
    """QLoRA settings. Set ``load_in_4bit: false`` for plain LoRA in bf16."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    # "auto" picks bf16 when the GPU supports it, otherwise fp16 (Colab T4).
    bnb_4bit_compute_dtype: str = "auto"


@dataclass
class LoraConf:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
    extra_target_modules: list[str] = field(default_factory=list)

    def resolved_targets(self) -> list[str]:
        seen: dict[str, None] = {}
        for name in [*self.target_modules, *self.extra_target_modules]:
            seen[name] = None
        return list(seen)


@dataclass
class DataConf:
    """Dataset location and shape.

    ``train_file`` may be a local file (jsonl/json/csv/parquet) or a Hugging Face
    dataset id. For a HF dataset, set ``dataset_name`` instead.
    """

    train_file: str | None = None
    eval_file: str | None = None
    dataset_name: str | None = None
    dataset_config: str | None = None
    dataset_split: str = "train"
    eval_split: str | None = None
    max_samples: int | None = None
    eval_samples: int | None = None

    # Vision track only: where relative image paths are resolved from.
    image_root: str | None = None
    max_image_tokens: int = 256

    max_length: int = 1024
    packing: bool = False
    # Train only on assistant tokens instead of the whole sequence.
    completion_only_loss: bool = True


@dataclass
class TrainingConf:
    output_dir: str = "outputs/lfm-run"
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int | None = None
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_strategy: str = "no"  # steps | epoch | no
    save_steps: int = 100
    save_total_limit: int | None = 2
    gradient_checkpointing: bool = True
    optim: str = "auto"  # auto -> paged_adamw_8bit when 4-bit, else adamw_torch
    seed: int = 42
    report_to: str | None = "none"
    bf16: str = "auto"  # auto | true | false
    max_grad_norm: float = 1.0


@dataclass
class AudioConf:
    """Audio track settings (consumed by the liquid-audio notebook, not by TRL).

    ``liquid_audio.trainer.Trainer`` does full fine-tuning -- it does not expose a
    LoRA switch -- so this track needs a bigger GPU than the QLoRA tracks.
    """

    system_prompt: str = "Perform ASR."
    # Where preprocess_dataset() writes its output.
    preprocessed_dir: str = "data/audio/train"
    context_length: int = 256
    batch_size: int = 8
    max_steps: int = 1000
    warmup_steps: int = 50
    lr: float = 1e-4
    num_workers: int = 2
    output_dir: str = "outputs/lfm25-audio"


@dataclass
class ExportConf:
    merge_adapter: bool = False
    merged_dir: str | None = None
    push_to_hub: bool = False
    hub_repo_id: str | None = None
    save_gguf: bool = False
    gguf_outfile: str | None = None
    gguf_quant: str = "q4_k_m"


@dataclass
class TrainConfig:
    model: ModelConf = field(default_factory=ModelConf)
    quant: QuantConf = field(default_factory=QuantConf)
    lora: LoraConf = field(default_factory=LoraConf)
    data: DataConf = field(default_factory=DataConf)
    audio: AudioConf = field(default_factory=AudioConf)
    training: TrainingConf = field(default_factory=TrainingConf)
    export: ExportConf = field(default_factory=ExportConf)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config section(s): {sorted(unknown)}")
        return cls(
            model=_build(ModelConf, raw.get("model")),
            quant=_build(QuantConf, raw.get("quant")),
            lora=_build(LoraConf, raw.get("lora")),
            data=_build(DataConf, raw.get("data")),
            audio=_build(AudioConf, raw.get("audio")),
            training=_build(TrainingConf, raw.get("training")),
            export=_build(ExportConf, raw.get("export")),
        )

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)


def _build(cls, values: dict[str, Any] | None):
    if values is None:
        return cls()
    valid = {f.name for f in fields(cls)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"unknown key(s) for {cls.__name__}: {sorted(unknown)}")
    return cls(**values)
