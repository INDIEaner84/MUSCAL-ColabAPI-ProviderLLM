#!/usr/bin/env python3
"""Score speech->function-call predictions the way Liquid's own recipe does.

Three layered metrics, each a subset of the previous one:

1. **format compliance** -- the output parses, the function is in the catalog
   and every required argument is present.
2. **function-name accuracy** -- ...and the function is the right one.
3. **argument accuracy** -- ...and every argument matches (case/whitespace
   normalised, numbers compared numerically).

Run it on the *unmodified* model first. Liquid's baseline result is 0% across
all three -- that floor is the point: it proves fine-tuning is required, not
just nice to have.

    python scripts/eval_toolcalls.py --gold data/eval.jsonl --pred preds.jsonl
    python scripts/eval_toolcalls.py --gold data/eval.jsonl --pred raw.txt --pred-raw
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# So the script works from anywhere: put the repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from muscal_agent.catalog import TOOLS_BY_NAME
from muscal_agent.toolcall import ToolCallError, normalise, parse_tool_call


def read_gold(path: Path) -> list[str]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        rows.append(obj["call"] if isinstance(obj, dict) else str(obj))
    return rows


def read_pred(path: Path, raw: bool, key: str) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    if raw:
        return [line.strip() for line in text.splitlines()]
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        out.append(obj[key] if isinstance(obj, dict) else str(obj))
    return out


def score(gold: list[str], pred: list[str], fmt: str = "pipe") -> dict[str, float]:
    if len(gold) != len(pred):
        raise SystemExit(f"length mismatch: gold={len(gold)} pred={len(pred)}")

    n = len(gold)
    fmt_ok = name_ok = args_ok = 0
    failures: list[dict[str, str]] = []

    for g, p in zip(gold, pred):
        try:
            gold_call = parse_tool_call(g, fmt=fmt)
        except ToolCallError as exc:
            raise SystemExit(f"gold file is malformed: {g!r} ({exc})")

        try:
            pred_call = parse_tool_call(p, fmt=fmt)
        except ToolCallError:
            failures.append({"gold": g, "pred": p, "why": "unparseable"})
            continue
        tool = TOOLS_BY_NAME.get(pred_call.name)
        if tool is None:
            failures.append({"gold": g, "pred": p, "why": "unknown function"})
            continue
        try:
            tool.validate(pred_call.args)
        except KeyError as exc:
            failures.append({"gold": g, "pred": p, "why": str(exc)})
            continue

        fmt_ok += 1
        if pred_call.name != gold_call.name:
            failures.append({"gold": g, "pred": p, "why": "wrong function"})
            continue
        name_ok += 1

        gold_args = {k: normalise(v) for k, v in gold_call.args.items()}
        pred_args = {k: normalise(v) for k, v in pred_call.args.items()}
        # Only compare arguments the gold target actually specifies.
        compared = {k: v for k, v in pred_args.items() if k in gold_args}
        if compared == gold_args:
            args_ok += 1
        else:
            failures.append({"gold": g, "pred": p, "why": "wrong arguments"})

    return {
        "samples": n,
        "format_compliance": fmt_ok,
        "function_name_accuracy": name_ok,
        "argument_accuracy": args_ok,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", required=True, help="jsonl with a `call` field per line")
    parser.add_argument("--pred", required=True, help="jsonl with a `call`/`prediction` field, or raw lines")
    parser.add_argument("--pred-raw", action="store_true", help="--pred is one raw prediction per line")
    parser.add_argument("--pred-key", default="call", help="field holding the prediction (default: call)")
    parser.add_argument("--fmt", default="pipe", choices=["pipe", "pythonic"])
    parser.add_argument("--show", type=int, default=8, help="how many failures to print")
    args = parser.parse_args()

    gold = read_gold(Path(args.gold))
    pred = read_pred(Path(args.pred), args.pred_raw, args.pred_key)
    result = score(gold, pred, fmt=args.fmt)

    n = result["samples"] or 1
    print(f"{'metric':<26}{'hit':>8}{'total':>8}{'pct':>9}")
    for key in ("format_compliance", "function_name_accuracy", "argument_accuracy"):
        hit = result[key]
        print(f"{key:<26}{hit:>8}{result['samples']:>8}{100 * hit / n:>8.1f}%")

    failures = result["failures"]
    if failures:
        print(f"\n{len(failures)} failure(s); first {min(args.show, len(failures))}:")
        for f in failures[: args.show]:
            print(f"  gold: {f['gold']}\n  pred: {f['pred']}\n  why : {f['why']}\n")


if __name__ == "__main__":
    main()
