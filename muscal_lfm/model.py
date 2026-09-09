"""Model / tokenizer / processor loading with Colab-friendly defaults.

Two things matter a lot on Colab and are handled here:

* **bf16 vs fp16** -- a free-tier T4 has no bf16 support, so the compute dtype
  and ``TrainingArguments`` must fall back to fp16 or bitsandbytes throws.
* **QLoRA preparation** -- 4-bit base weights need ``prepare_model_for_kbit_training``
  plus ``enable_input_require_grads`` or gradient checkpointing silently produces
  zero gradients.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .config import TrainConfig


def supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    return bool(torch.cuda.is_bf16_supported())


def gpu_info() -> dict[str, Any]:
    """Small report used by the notebooks to pick sane batch sizes."""
    if not torch.cuda.is_available():
        return {"gpu": None, "vram_gb": 0.0}
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return {"gpu": name, "vram_gb": round(total_gb, 1), "bf16": supports_bf16()}


def resolve_dtype(pref: str = "auto") -> torch.dtype:
    if pref == "bf16":
        return torch.bfloat16
    if pref == "fp16":
        return torch.float16
    # auto
    return torch.bfloat16 if supports_bf16() else torch.float16


def build_bnb_config(cfg: TrainConfig) -> BitsAndBytesConfig | None:
    q = cfg.quant
    if not q.load_in_4bit:
        return None
    compute_dtype = (
        resolve_dtype("auto") if q.bnb_4bit_compute_dtype == "auto" else resolve_dtype(q.bnb_4bit_compute_dtype)
    )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=q.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=q.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_tokenizer(cfg: TrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.id, trust_remote_code=cfg.model.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_processor(cfg: TrainConfig):
    """VL track: the processor owns both the tokenizer and the image transform."""
    try:
        processor = AutoProcessor.from_pretrained(
            cfg.model.id,
            max_image_tokens=cfg.data.max_image_tokens,
            trust_remote_code=cfg.model.trust_remote_code,
        )
    except TypeError:
        # Older/newer processors without the max_image_tokens knob.
        processor = AutoProcessor.from_pretrained(
            cfg.model.id, trust_remote_code=cfg.model.trust_remote_code
        )
    if getattr(processor, "tokenizer", None) is not None and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def load_model(cfg: TrainConfig):
    """Load the base checkpoint for the configured track."""
    quantization_config = build_bnb_config(cfg)
    dtype = resolve_dtype("auto")
    kwargs: dict[str, Any] = dict(
        dtype=dtype,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if cfg.model.attn_implementation:
        kwargs["attn_implementation"] = cfg.model.attn_implementation

    if cfg.model.track == "vl":
        return AutoModelForImageTextToText.from_pretrained(cfg.model.id, **kwargs)
    if cfg.model.track in {"text", "moe"}:
        return AutoModelForCausalLM.from_pretrained(cfg.model.id, **kwargs)
    raise ValueError(
        f"track {cfg.model.track!r} is not handled here (audio uses liquid-audio's own trainer)"
    )


def prepare_for_training(model: nn.Module, cfg: TrainConfig) -> nn.Module:
    """Apply k-bit training prep + gradient checkpointing.

    Only needed when the base weights are quantised (QLoRA) or when checkpointing
    is requested; for a plain bf16 LoRA run with enough VRAM this is a no-op.
    """
    if cfg.quant.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.training.gradient_checkpointing
        )

    if cfg.training.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model


def apply_lora(model: nn.Module, cfg: TrainConfig):
    from peft import LoraConfig, get_peft_model

    targets = cfg.lora.resolved_targets()
    peft_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    _warn_missing_targets(model, targets)
    return model


def _warn_missing_targets(model: nn.Module, targets: list[str]) -> None:
    matched = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = [t for t in targets if t not in matched]
    if missing:
        print(
            "[warn] LoRA target module(s) not found in this checkpoint: "
            f"{missing}. Found modules like: {sorted(matched)[:12]} ..."
        )


def print_trainable_parameters(model: nn.Module) -> None:
    trainable, total = 0, 0
    for param in model.parameters():
        numel = param.numel()
        total += numel
        if param.requires_grad:
            trainable += numel
    pct = 100 * trainable / total if total else 0.0
    print(f"trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
