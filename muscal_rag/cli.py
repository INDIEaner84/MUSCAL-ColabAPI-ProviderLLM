"""CLI for the RAG baseline.

    python -m muscal_rag build  --config configs/rag_baseline.yaml
    python -m muscal_rag search "Wie gross ist der Chunk?" --config configs/rag_baseline.yaml
    python -m muscal_rag eval   --config configs/rag_baseline.yaml
    python -m muscal_rag eval   --config configs/rag_baseline.yaml --backend hashing   # pipeline test
"""

from __future__ import annotations

import argparse
import json

from .config import RagConfig
from .metrics import load_eval_cases
from .pipeline import build_index, evaluate, generate_answer, search


def main() -> None:
    parser = argparse.ArgumentParser(description="LFM RAG baseline: build, search, evaluate")
    parser.add_argument("command", choices=["build", "search", "eval", "answer"])
    parser.add_argument("--config", default="configs/rag_baseline.yaml")
    parser.add_argument("query", nargs="?", help="question for `search` / `answer`")
    parser.add_argument("--backend", choices=["sentence-transformers", "hashing"],
                        help="override the embedding backend")
    parser.add_argument("--top-k", type=int, help="override retrieval.top_k")
    parser.add_argument("--eval-file", help="override eval.eval_file")
    parser.add_argument("--show-misses", type=int, help="how many zero-recall questions to print")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    cfg = RagConfig.from_yaml(args.config)
    if args.backend:
        cfg.embedding.backend = args.backend
    if args.top_k:
        cfg.retrieval.top_k = args.top_k
    if args.eval_file:
        cfg.eval.eval_file = args.eval_file

    if args.command == "build":
        index = build_index(cfg)
        print(json.dumps(index.meta, indent=2))
        return

    if args.command in {"search", "answer"}:
        if not args.query:
            raise SystemExit("a question is required")
        hits = search(cfg, [args.query])[0]
        if args.command == "search":
            for hit in hits:
                print(f"{hit.score:6.3f}  {hit.chunk.ref:<28} {hit.chunk.text[:90].replace(chr(10), ' ')}")
            return
        print(generate_answer(cfg, args.query, hits))
        return

    if args.command == "eval":
        cases = load_eval_cases(cfg.eval.eval_file)
        report, _ = evaluate(cfg, cases)
        if args.json:
            print(json.dumps({
                "n": report.n,
                "recall_at_k": report.recall_at_k,
                "mrr": report.mrr,
                "hit_rate": report.hit_rate,
                "backend": cfg.embedding.backend,
            }, indent=2))
            return
        print(report.as_text(show_misses=args.show_misses or cfg.eval.show_misses))
        if cfg.embedding.backend == "hashing":
            print("\n[warn] hashing-Backend: das prüft die Pipeline, nicht die Qualität.")
            print("       Für echte Zahlen: --backend sentence-transformers")
        else:
            print(f"\n[embedder] {cfg.embedding.model}")
        return


if __name__ == "__main__":
    main()
