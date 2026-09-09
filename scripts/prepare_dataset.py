#!/usr/bin/env python3
"""Turn "whatever you have" into the JSONL the trainer expects.

Run this locally (no GPU needed) before uploading to Drive:

    python scripts/prepare_dataset.py --input faq.csv --format qa --out data/train.jsonl
    python scripts/prepare_dataset.py --input data/train.jsonl --validate-only

Output format (one JSON object per line):

    {"messages": [{"role": "system", "content": "..."},
                  {"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

VALID_ROLES = {"system", "user", "assistant"}


def read_rows(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get("data", [data])
    raise SystemExit(f"unsupported input type: {suffix} (use csv, json or jsonl)")


def detect_format(rows: list[dict]) -> str:
    keys = {k.lower() for k in rows[0]} if rows else set()
    if "messages" in keys or "conversations" in keys:
        return "sharegpt" if "conversations" in keys else "messages"
    if {"instruction", "output"} <= keys:
        return "alpaca"
    if {"question", "answer"} <= keys:
        return "qa"
    if {"image", "answer"} <= keys:
        return "flat-vl"
    if {"prompt", "completion"} <= keys:
        return "prompt-completion"
    return "unknown"


ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "ai": "assistant",
    "bot": "assistant",
}


def convert(rows: list[dict], fmt: str, *, system_prompt: str | None) -> list[dict]:
    out: list[dict] = []

    for i, row in enumerate(rows):
        lower = {str(k).lower(): v for k, v in row.items()}

        if fmt == "messages":
            messages = row.get("messages") or []
        elif fmt == "sharegpt":
            messages = [
                {"role": ROLE_ALIASES.get(str(m.get("from", "")).lower(), "user"),
                 "content": m.get("value", "")}
                for m in row.get("conversations", [])
            ]
        elif fmt == "alpaca":
            instruction = str(lower.get("instruction", "")).strip()
            inp = str(lower.get("input", "")).strip()
            user = f"{instruction}\n\n{inp}".strip() if inp else instruction
            messages = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": str(lower.get("output", "")).strip()},
            ]
        elif fmt == "qa":
            messages = [
                {"role": "user", "content": str(lower.get("question", "")).strip()},
                {"role": "assistant", "content": str(lower.get("answer", "")).strip()},
            ]
        elif fmt == "prompt-completion":
            messages = [
                {"role": "user", "content": str(lower.get("prompt", "")).strip()},
                {"role": "assistant", "content": str(lower.get("completion", "")).strip()},
            ]
        elif fmt == "flat-vl":
            # Vision datasets stay flat; the trainer resolves images itself.
            out.append(row)
            continue
        else:
            raise SystemExit(f"row {i}: cannot infer format, pass --format explicitly")

        messages = [m for m in messages if str(m.get("content", "")).strip()]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        out.append({"messages": messages})

    return out


def validate(rows: list[dict], *, flat_vl: bool = False) -> list[str]:
    problems: list[str] = []
    if flat_vl:
        for i, row in enumerate(rows):
            for key in ("image", "question", "answer"):
                if key not in row:
                    problems.append(f"row {i}: missing column {key!r}")
        return problems

    for i, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            problems.append(f"row {i}: missing/empty `messages`")
            continue
        for msg in messages:
            if msg.get("role") not in VALID_ROLES:
                problems.append(f"row {i}: bad role {msg.get('role')!r}")
        if messages[-1].get("role") != "assistant":
            problems.append(f"row {i}: last message must be from `assistant`")
        if sum(1 for m in messages if m.get("role") == "assistant") == 0:
            problems.append(f"row {i}: no assistant message")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="csv / json / jsonl")
    parser.add_argument("--format", default="auto",
                        choices=["auto", "messages", "sharegpt", "alpaca", "qa", "prompt-completion", "flat-vl"])
    parser.add_argument("--out", default="data/train.jsonl")
    parser.add_argument("--system", default=None, help="prepend this system prompt to every row")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    path = Path(args.input)
    rows = read_rows(path)
    if args.max_rows:
        rows = rows[: args.max_rows]

    fmt = detect_format(rows) if args.format == "auto" else args.format
    if fmt in {"unknown", "auto"}:
        sys.exit(f"could not infer the format of {path}; columns found: {list(rows[0]) if rows else []}")

    if args.format == "auto":
        print(f"[info] detected format: {fmt}")

    converted = convert(rows, fmt, system_prompt=args.system)
    problems = validate(converted, flat_vl=fmt == "flat-vl")

    if problems:
        print(f"[warn] {len(problems)} problem(s); showing first 10:")
        for p in problems[:10]:
            print("  -", p)
    else:
        print(f"[ok] {len(converted)} rows look valid")

    if args.validate_only:
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in converted:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[write] {len(converted)} rows -> {out_path}")


if __name__ == "__main__":
    main()
