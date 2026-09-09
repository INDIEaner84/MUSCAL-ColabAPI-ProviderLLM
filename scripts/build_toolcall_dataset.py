#!/usr/bin/env python3
"""Build an (audio, function call) dataset for the audio agent fine-tune.

Input is cheap: a JSONL of spoken commands and the function call they should
produce.

    {"utterance": "such im netz nach liquid ai lfm", "call": "web_search|query=liquid ai lfm"}
    {"utterance": "oeffne den browser auf example.com", "call": "browser_open|url=https://example.com"}

Two ways to get the audio:

**``--audio-map`` (recommended for quality)** -- you supply WAVs (your own
recordings, a TTS service, whatever). Map file lines::

    {"utterance": "such im netz nach liquid ai lfm", "audio": "wav/0001.wav"}

**``--tts``** -- synthesise the utterances with LFM2.5-Audio itself (needs GPU
+ ``liquid-audio``). Convenient, but every sample has the same voice, which is
exactly the kind of uniformity that makes a model brittle to *your* voice.
Mix in real recordings whenever you can.

Output matches the OHF-Voice schema so the standard preprocess script works
unchanged: a Hugging Face dataset with an ``audio_chat`` column.

    python scripts/build_toolcall_dataset.py \
        --utterances data/utterances.jsonl --out data/agent_audio --tts

    python scripts/build_toolcall_dataset.py \
        --utterances data/utterances.jsonl --audio-map data/audio_map.jsonl --out data/agent_audio
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

# So the script works from anywhere: put the repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Built-in paraphrase bank. Templates are keyed by tool name and use the tool's
# argument names as placeholders. Extend it with --templates (same JSON shape).
DEFAULT_TEMPLATES: dict[str, list[str]] = {
    "web_search": [
        "such nach {query}",
        "such im netz nach {query}",
        "recherchiere {query}",
        "finde informationen zu {query}",
        "google {query}",
    ],
    "web_read": [
        "lies die seite {url}",
        "was steht auf {url}",
        "oeffne und lies {url}",
        "fass den inhalt von {url} zusammen",
    ],
    "browser_open": [
        "oeffne {url} im browser",
        "geh zu {url}",
        "browser auf {url}",
        "start den browser mit {url}",
    ],
    "browser_click": [
        "klick auf {selector}",
        "drueck {selector}",
        "klick den button {selector}",
    ],
    "browser_type": [
        "tippe {text} in {selector}",
        "schreib {text} in {selector}",
        "gib {text} bei {selector} ein",
    ],
    "desktop_launch": [
        "starte {app}",
        "oeffne {app}",
        "mach {app} auf",
        "start {app}",
    ],
    "desktop_type": [
        "schreib {text}",
        "tippe {text}",
        "gib {text} ein",
    ],
    "desktop_hotkey": [
        "drueck {keys}",
        "tastenkuerzel {keys}",
        "mach {keys}",
    ],
    "desktop_click": [
        "klick auf {x} {y}",
        "klick bei {x} {y}",
    ],
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def synthesise_pairs(
    slots: dict[str, dict[str, list]],
    templates: dict[str, list[str]],
    per_tool: int,
    fmt: str = "pipe",
    seed: int = 42,
) -> list[tuple[str, str]]:
    """Generate utterance/call pairs for EVERY tool in the catalog.

    Builds the full (templates x slot-value-combinations) product per tool,
    shuffles it deterministically and takes the first `per_tool` unique pairs.

    Enumerating the product matters: cycling the lists in lockstep collapses to
    lcm(len(values), len(templates)) combinations, which silently starves
    single-argument tools (12 urls x 6 templates yielded only 12 pairs).
    """
    import itertools
    import random

    from muscal_agent.catalog import TOOLS_BY_NAME
    from muscal_agent.toolcall import ToolCall, ToolCallError

    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []

    for name in sorted(TOOLS_BY_NAME):
        tool = TOOLS_BY_NAME[name]
        tmpl_list = templates.get(name) or []
        if not tmpl_list:
            continue

        arg_names = [p.name for p in tool.params if slots.get(name, {}).get(p.name)]
        if arg_names:
            combos = [
                dict(zip(arg_names, values))
                for values in itertools.product(*(slots[name][a] for a in arg_names))
            ]
        else:
            combos = [{}]

        candidates = list(itertools.product(tmpl_list, combos))
        rng.shuffle(candidates)

        seen: set[tuple[str, str]] = set()
        for template, values in candidates:
            if len(seen) >= per_tool:
                break
            values = dict(values)
            # A template promising "and press enter" must not emit submit=false.
            lowered = template.lower()
            if "submit" in values:
                if "press enter" in lowered or "sende ab" in lowered:
                    values["submit"] = "true"
                else:
                    values["submit"] = "false"
            try:
                text = template.format(**values)
                call = ToolCall(name, values).serialise(fmt)
            except (KeyError, ToolCallError):
                continue
            if (text, call) not in seen:
                seen.add((text, call))
                pairs.append((text, call))
    return pairs


def augment(items: list[tuple[str, str]], templates: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Generate paraphrases from the tool arguments of each call."""
    from muscal_agent.toolcall import ToolCallError, parse_pipe

    out: list[tuple[str, str]] = []
    for utterance, call in items:
        out.append((utterance, call))
        try:
            parsed = parse_pipe(call)
        except ToolCallError:
            continue
        for template in templates.get(parsed.name, []):
            try:
                text = template.format(**{k: v for k, v in parsed.args.items()})
            except KeyError:
                continue
            if text != utterance:
                out.append((text, call))
    return out


def wav_bytes(path: Path) -> bytes:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV")
    return buf.getvalue()


def synthesise(utterances: list[str], *, model_id: str, voice_prompt: str, out_dir: Path) -> list[bytes]:
    """TTS with LFM2.5-Audio. One voice for everything -- see module docstring."""
    import torch
    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

    processor = LFM2AudioProcessor.from_pretrained(model_id, device="cuda").eval()
    model = LFM2AudioModel.from_pretrained(model_id).eval().cuda()

    blobs: list[bytes] = []
    for text in utterances:
        chat = ChatState(processor)
        chat.new_turn("system")
        chat.add_text(voice_prompt)
        chat.end_turn()
        chat.new_turn("user")
        chat.add_text(text)
        chat.end_turn()
        chat.new_turn("assistant")

        audio_out = []
        for token in model.generate_sequential(**chat, max_new_tokens=512):
            if token.numel() > 1:
                audio_out.append(token)
        codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
        waveform = processor.decode(codes).cpu()[0]

        buf = io.BytesIO()
        import soundfile as sf

        sf.write(buf, waveform.numpy(), 24_000, format="WAV")
        blobs.append(buf.getvalue())
    return blobs


def build_rows(pairs: list[tuple[str, str]], audios: list[bytes], system_prompt: str) -> list[dict]:
    rows = []
    for (utterance, call), audio in zip(pairs, audios):
        rows.append(
            {
                "utterance": utterance,
                "call": call,
                "audio_chat": [
                    {"role": "system", "content": [{"modality": "text", "text": system_prompt}]},
                    {"role": "user", "content": [{"modality": "audio", "audio": audio}]},
                    {"role": "assistant", "content": [{"modality": "text", "text": call}]},
                ],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--utterances", help="jsonl: {utterance, call}")
    parser.add_argument("--out", default="data/agent_audio")
    parser.add_argument("--audio-map", help="jsonl: {utterance, audio} -> use these WAVs")
    parser.add_argument("--tts", action="store_true", help="synthesise audio with LFM2.5-Audio")
    parser.add_argument("--tts-model", default="LiquidAI/LFM2.5-Audio-1.5B")
    parser.add_argument("--voice", default="Perform TTS. Use the US female voice.")
    parser.add_argument("--system-prompt", default="Perform ASR.",
                        help="must match the runtime; see docs/audio-agent.md")
    parser.add_argument("--augment", action="store_true", help="add paraphrases from the template bank")
    parser.add_argument("--templates", help="json file: {tool_name: [templates]} to extend/override")
    parser.add_argument("--slots", help="json: {tool: {arg: [values]}} to synthesise pairs")
    parser.add_argument("--per-tool", type=int, default=60,
                        help="how many pairs to synthesise per tool (default: 60)")
    parser.add_argument("--language", choices=["de", "en"], default="de",
                        help="bundled template + slot files from data/ (default: de)")
    parser.add_argument("--fmt", default="pipe", choices=["pipe", "pythonic"],
                        help="tool-call wire format (must match the runtime)")
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="held-out share, stratified by function name (0 disables)")
    parser.add_argument("--eval-out", default="data/agent_eval.jsonl",
                        help="where the gold calls of the val split are written")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-pairs", help="dump the generated utterance/call pairs here and stop")
    parser.add_argument("--push-to", help="push the dataset to this HF repo id")
    args = parser.parse_args()

    items: list[tuple[str, str]] = []

    templates = dict(DEFAULT_TEMPLATES)
    bundled_templates = Path(f"data/agent_templates_{args.language}.json")
    if bundled_templates.exists():
        templates.update(json.loads(bundled_templates.read_text(encoding="utf-8")))
    if args.templates:
        templates.update(json.loads(Path(args.templates).read_text(encoding="utf-8")))

    if args.slots:
        slots = json.loads(Path(args.slots).read_text(encoding="utf-8"))
        synth = synthesise_pairs(slots, templates, args.per_tool, fmt=args.fmt)
        print(f"[synth] {len(synth)} pairs across all tools ({args.language}, {args.per_tool}/tool)")
        items.extend(synth)

    if args.utterances:
        hand = [(r["utterance"], r["call"]) for r in load_jsonl(Path(args.utterances))]
        print(f"[input] {len(hand)} hand-written pairs")
        items.extend(hand)

    if not items:
        raise SystemExit("no data: pass --utterances and/or --slots")

    # Deduplicate on the exact (utterance, call) pair, order preserved.
    seen: set[tuple[str, str]] = set()
    unique = []
    for pair in items:
        if pair not in seen:
            seen.add(pair)
            unique.append(pair)
    items = unique
    print(f"[total] {len(items)} unique pairs")
    if args.augment:
        before = len(items)
        items = augment(items, templates)
        print(f"[augment] {before} -> {len(items)} pairs")

    if args.write_pairs:
        from collections import Counter
        counts = Counter(c.split("|", 1)[0] for _, c in items)
        print("[coverage] " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

        out_pairs = Path(args.write_pairs)
        out_pairs.parent.mkdir(parents=True, exist_ok=True)
        out_pairs.write_text(
            "\n".join(json.dumps({"utterance": u, "call": c}, ensure_ascii=False) for u, c in items) + "\n",
            encoding="utf-8",
        )
        print(f"[write] {len(items)} pairs -> {out_pairs}")
        if not (args.audio_map or args.tts):
            return

    if args.audio_map:
        mapping = {r["utterance"]: r["audio"] for r in load_jsonl(Path(args.audio_map))}
        missing = [u for u, _ in items if u not in mapping]
        if missing:
            raise SystemExit(f"{len(missing)} utterance(s) have no audio, e.g. {missing[:3]}")
        audios = [wav_bytes(Path(mapping[u])) for u, _ in items]
        print(f"[audio] {len(audios)} existing WAVs")
    elif args.tts:
        audios = synthesise([u for u, _ in items], model_id=args.tts_model,
                            voice_prompt=args.voice, out_dir=Path(args.out))
        print(f"[audio] synthesised {len(audios)} clips")
    else:
        raise SystemExit("choose one audio source: --audio-map or --tts")

    from datasets import Dataset

    rows = build_rows(items, audios, args.system_prompt)
    out = Path(args.out)

    if args.val_ratio > 0:
        train_rows, val_rows = stratified_split(rows, args.val_ratio, seed=args.seed)
        print(f"[split] train={len(train_rows)} val={len(val_rows)}")

        val_dir = out.with_name(out.name + "_val")
        Dataset.from_list(val_rows).save_to_disk(str(val_dir))
        print(f"[write] val -> {val_dir}")

        eval_out = Path(args.eval_out)
        eval_out.parent.mkdir(parents=True, exist_ok=True)
        eval_out.write_text(
            "\n".join(json.dumps({"call": r["call"]}, ensure_ascii=False) for r in val_rows) + "\n",
            encoding="utf-8",
        )
        print(f"[write] gold calls -> {eval_out}")
    else:
        train_rows = rows

    out.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list(train_rows)
    ds.save_to_disk(str(out))
    print(f"[write] {len(ds)} rows -> {out}")

    if args.push_to:
        ds.push_to_hub(args.push_to)
        print(f"[push] -> {args.push_to}")


def stratified_split(rows: list[dict], ratio: float, *, seed: int = 42):
    """Hold out `ratio` of the samples *per function*, never at random.

    A random split leaves rare functions with zero eval coverage, which makes
    the metric look better than the model is.
    """
    import random

    rng = random.Random(seed)
    by_fn: dict[str, list[dict]] = {}
    for row in rows:
        fn = row["call"].split("|", 1)[0]
        by_fn.setdefault(fn, []).append(row)

    train, val = [], []
    for fn in sorted(by_fn):
        group = by_fn[fn][:]
        rng.shuffle(group)
        n_val = max(1, round(len(group) * ratio)) if len(group) > 1 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


if __name__ == "__main__":
    main()
