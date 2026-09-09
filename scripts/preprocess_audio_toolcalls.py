#!/usr/bin/env python3
"""Preprocess an (audio, function call) dataset into liquid-audio's tensor format.

Adapted from Liquid's reference script
(cookbook/examples/voice-assistant/scripts/preprocess_ohf_voice.py).

    python scripts/preprocess_audio_toolcalls.py \
        --dataset data/agent_audio --output-path data/agent_audio/train

    # or straight from the Hub / from the published reference dataset:
    python scripts/preprocess_audio_toolcalls.py --dataset Paulescu/OHF-Voice-audio-20260504 --split train
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator

from datasets import load_dataset, load_from_disk

from liquid_audio import LFM2AudioProcessor
from liquid_audio.data.mapper import LFM2AudioChatMapper
from liquid_audio.data.preprocess import preprocess_dataset
from liquid_audio.data.types import AudioSegment, ChatMessage, TextSegment

# Do not change this without reading docs/audio-agent.md first.
#
# llama-liquid-audio-server only accepts a closed allow-list of system prompts
# ("Perform ASR.", the "Perform TTS. ..." variants, and the interleaved one) and
# forces one at inference time. If we train with a different prompt, the runtime
# pushes the model back into transcription mode and the fine-tune is silently
# overridden. Liquid verified this empirically: correct calls in PyTorch,
# plain transcription through the GGUF server. So we train on the exact string
# the server will send.
SYSTEM_PROMPT = "Perform ASR."


class ToolCallIterator:
    """Yields one chat per dataset row, in the shape the mapper expects."""

    def __init__(self, rows: Iterator[dict], system_prompt: str = SYSTEM_PROMPT) -> None:
        self.rows = rows
        self.system_prompt = system_prompt

    def __iter__(self) -> Iterator[list[ChatMessage]]:
        for row in self.rows:
            messages: list[ChatMessage] = [
                ChatMessage(role="system", content=[TextSegment(text=self.system_prompt)])
            ]
            for msg in row["audio_chat"]:
                if msg["role"] == "system":
                    continue  # we inject our own, pinned prompt
                segments = []
                for item in msg.get("content", []):
                    if item.get("modality") == "audio" and item.get("audio"):
                        segments.append(AudioSegment(audio=item["audio"]))
                    elif item.get("modality") == "text" and item.get("text"):
                        segments.append(TextSegment(text=item["text"]))
                if segments:
                    messages.append(ChatMessage(role=msg["role"], content=segments))
            if len(messages) > 1:
                yield messages


def load_rows(dataset: str, split: str | None):
    path = Pathish(dataset)
    if path.exists():
        ds = load_from_disk(dataset)
        if hasattr(ds, "keys"):  # DatasetDict
            ds = ds[split or "train"]
    else:
        ds = load_dataset(dataset, split=split or "train")
    return ds


def Pathish(value: str):
    from pathlib import Path

    return Path(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="data/agent_audio",
                        help="local dataset dir (save_to_disk) or HF repo id")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-path", default="data/agent_audio/train")
    parser.add_argument("--max-context-length", type=int, default=512,
                        help="skip samples longer than this many tokens")
    parser.add_argument("--model", default="LiquidAI/LFM2.5-Audio-1.5B")
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    args = parser.parse_args()

    ds = load_rows(args.dataset, args.split)
    print(f"[data] {len(ds)} rows from {args.dataset}")

    processor = LFM2AudioProcessor.from_pretrained(args.model, device=args.device).eval()
    mapper = LFM2AudioChatMapper(processor)
    preprocess_dataset(
        data=ToolCallIterator(ds),
        output_path=args.output_path,
        mapper=mapper,
        max_context_length=args.max_context_length,
    )
    print(f"[write] preprocessed dataset -> {args.output_path}")


if __name__ == "__main__":
    main()
