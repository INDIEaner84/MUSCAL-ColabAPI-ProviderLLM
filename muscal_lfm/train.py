"""TRL-based SFT for the text, MoE and vision tracks.

The three tracks differ only in (a) which Auto class loads the model, (b) whether
the dataset is a HF ``Dataset`` of conversations or a list of VL conversations
with a custom collator. Everything else is shared.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from . import data as data_utils
from .config import TrainConfig
from .model import apply_lora, load_model, load_processor, load_tokenizer, prepare_for_training, print_trainable_parameters, supports_bf16


def _filter_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only keys the (TRL-version-dependent) config dataclass understands."""
    valid = {f.name for f in dataclasses.fields(cls)}
    dropped = {k: v for k, v in kwargs.items() if k not in valid}
    if dropped:
        print(f"[info] ignoring unsupported SFTConfig keys for this TRL version: {sorted(dropped)}")
    return {k: v for k, v in kwargs.items() if k in valid}


def _resolve_precision(cfg: TrainConfig) -> tuple[bool, bool]:
    """Return (bf16, fp16) for TrainingArguments."""
    pref = (cfg.training.bf16 or "auto").lower()
    if pref == "true":
        return True, False
    if pref == "false":
        return False, True
    bf16 = supports_bf16()
    return bf16, not bf16


def _resolve_optim(cfg: TrainConfig) -> str:
    if cfg.training.optim != "auto":
        return cfg.training.optim
    return "paged_adamw_8bit" if cfg.quant.load_in_4bit else "adamw_torch"


def build_sft_config(cfg: TrainConfig, **overrides) -> Any:
    from trl import SFTConfig

    bf16, fp16 = _resolve_precision(cfg)
    t = cfg.training
    kwargs: dict[str, Any] = dict(
        output_dir=t.output_dir,
        num_train_epochs=t.num_train_epochs,
        max_steps=t.max_steps,
        per_device_train_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler_type,
        warmup_ratio=t.warmup_ratio,
        weight_decay=t.weight_decay,
        logging_steps=t.logging_steps,
        save_strategy=t.save_strategy,
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        gradient_checkpointing=t.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=_resolve_optim(cfg),
        max_grad_norm=t.max_grad_norm,
        seed=t.seed,
        report_to=t.report_to,
        bf16=bf16,
        fp16=fp16,
        max_length=cfg.data.max_length,
        packing=cfg.data.packing,
    )
    if t.per_device_eval_batch_size:
        kwargs["per_device_eval_batch_size"] = t.per_device_eval_batch_size
    # TRL renamed assistant_only_loss -> completion_only_loss; support both.
    if cfg.data.completion_only_loss:
        kwargs["completion_only_loss"] = True
        kwargs["assistant_only_loss"] = True
    kwargs.update(overrides)
    return SFTConfig(**_filter_kwargs(SFTConfig, kwargs))


def build_vl_collator(processor):
    """Multimodal collator: apply the chat template + mask pad tokens in labels."""

    def collate_fn(samples):
        batch = processor.apply_chat_template(
            samples, tokenize=True, return_dict=True, return_tensors="pt"
        )
        labels = batch["input_ids"].clone()
        pad_id = processor.tokenizer.pad_token_id
        labels[labels == pad_id] = -100
        batch["labels"] = labels
        return batch

    return collate_fn


def build_trainer(cfg: TrainConfig):
    """Assemble model + data + trainer for the configured track."""
    from trl import SFTTrainer

    track = cfg.model.track
    if track in {"text", "moe"}:
        tokenizer = load_tokenizer(cfg)
        train_ds = data_utils.load_split(cfg)
        data_utils.validate_conversational(train_ds)
        eval_ds = data_utils.load_eval_split(cfg)
        if eval_ds is not None:
            data_utils.validate_conversational(eval_ds)

        model = load_model(cfg)
        model = prepare_for_training(model, cfg)
        model = apply_lora(model, cfg)
        print_trainable_parameters(model)

        report = data_utils.dataset_report(train_ds)
        print(f"[data] {report}")

        trainer = SFTTrainer(
            model=model,
            args=build_sft_config(cfg),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            peft_config=None,  # adapter already applied so we control target modules
        )
        return trainer, tokenizer

    if track == "vl":
        processor = load_processor(cfg)
        raw_ds = data_utils.load_split(cfg)
        train_conversations = data_utils.build_vlm_conversations(
            raw_ds, image_root=cfg.data.image_root
        )
        print(f"[data] {len(train_conversations)} vision conversations")

        model = load_model(cfg)
        model = prepare_for_training(model, cfg)
        model = apply_lora(model, cfg)
        print_trainable_parameters(model)

        trainer = SFTTrainer(
            model=model,
            args=build_sft_config(cfg, dataset_kwargs={"skip_prepare_dataset": True}),
            train_dataset=train_conversations,
            data_collator=build_vl_collator(processor),
            processing_class=processor.tokenizer,
            peft_config=None,
        )
        return trainer, processor.tokenizer

    raise ValueError(
        f"track {track!r} has no TRL trainer here -- use the liquid-audio notebook for audio"
    )


def train(cfg: TrainConfig) -> Path:
    """Run training and save adapter + tokenizer. Returns the adapter directory."""
    trainer, tokenizer = build_trainer(cfg)
    trainer.train()

    adapter_dir = Path(cfg.training.output_dir) / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[save] adapter -> {adapter_dir}")

    metrics = trainer.evaluate() if trainer.eval_dataset is not None else None
    if metrics:
        print(f"[eval] {metrics}")

    return adapter_dir
