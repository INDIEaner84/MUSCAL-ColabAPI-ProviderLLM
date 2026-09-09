#!/usr/bin/env python3
"""Merge a trained LoRA adapter back into its base model.

    python scripts/merge_adapter.py \
        --base LiquidAI/LFM2.5-1.2B-Instruct \
        --adapter outputs/lfm25-1.2b-text/adapter \
        --out outputs/lfm25-1.2b-text/merged

Merging runs on CPU on purpose: a 4-bit quantised training model must not be
merged, and CPU avoids the VRAM spike that would kill a 15 GB T4.
"""

from __future__ import annotations

import argparse

from muscal_lfm.export import merge_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="base model id, e.g. LiquidAI/LFM2.5-1.2B-Instruct")
    parser.add_argument("--adapter", required=True, help="directory with adapter_model.safetensors")
    parser.add_argument("--out", required=True)
    parser.add_argument("--track", default="text", choices=["text", "moe", "vl"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16"])
    args = parser.parse_args()

    merge_adapter(
        args.base,
        args.adapter,
        args.out,
        track=args.track,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
