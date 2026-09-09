"""Command line entry point: ``python -m muscal_lfm --config configs/text_qlora.yaml``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import TrainConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="muscal_lfm",
        description="LoRA / QLoRA fine-tuning for Liquid AI LFM2.5 models (Colab-friendly).",
    )
    parser.add_argument("--config", required=True, help="path to a YAML run config")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. training.learning_rate=1e-4")
    parser.add_argument("--merge", action="store_true", help="merge the adapter after training")
    parser.add_argument("--gguf", action="store_true", help="export GGUF after merging")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved config and exit (no model download)")
    return parser


def apply_overrides(cfg: TrainConfig, overrides: list[str]) -> TrainConfig:
    import yaml

    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"invalid override {item!r}, expected section.key=value")
        key, raw_value = item.split("=", 1)
        section, field = key.split(".", 1)
        target = getattr(cfg, section, None)
        if target is None:
            raise SystemExit(f"unknown config section {section!r}")
        if not hasattr(target, field):
            raise SystemExit(f"unknown key {key!r}")
        current = getattr(target, field)
        value = yaml.safe_load(raw_value)
        if isinstance(current, bool):
            value = str(value).lower() in {"true", "1", "yes"}
        elif isinstance(current, int) and not isinstance(current, bool):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        setattr(target, field, value)
    return cfg


def main() -> None:
    args = build_parser().parse_args()
    cfg = apply_overrides(TrainConfig.from_yaml(args.config), args.set)

    if args.merge:
        cfg.export.merge_adapter = True
    if args.gguf:
        cfg.export.save_gguf = True
        cfg.export.merge_adapter = True

    # --dry-run must work without torch/transformers installed (local data prep box).
    if args.dry_run:
        print(f"# resolved config ({args.config})\n{cfg.to_yaml()}")
        return

    from .model import gpu_info
    from .train import train

    print(f"[gpu] {gpu_info()}")
    Path(cfg.training.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.training.output_dir) / "config.resolved.yaml").write_text(
        cfg.to_yaml(), encoding="utf-8"
    )

    adapter_dir = train(cfg)

    from .export import run_export

    run_export(cfg, adapter_dir)


if __name__ == "__main__":
    main()
