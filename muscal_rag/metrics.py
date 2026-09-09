"""Retrieval metrics: recall@k, MRR, hit rate.

These measure the retriever, not the generator -- and that is the point. If the
right passage is not in the top-k, no amount of fine-tuning the reader will make
the answer correct. Measure retrieval first, then decide whether an adapter is
needed at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .index import SearchHit


@dataclass
class EvalCase:
    question: str
    relevant: list[str]  # chunk refs ("doc#3") or bare doc ids ("doc")
    note: str = ""


@dataclass
class CaseResult:
    question: str
    relevant: list[str]
    retrieved: list[str]
    recall_at_k: dict[int, float]
    reciprocal_rank: float
    hit: bool


@dataclass
class EvalReport:
    ks: list[int]
    n: int
    recall_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    hit_rate: float = 0.0
    misses: list[CaseResult] = field(default_factory=list)

    def as_text(self, show_misses: int = 5) -> str:
        lines = [f"Fragen: {self.n}", ""]
        lines.append(f"{'k':>4}  {'recall@k':>10}")
        for k in self.ks:
            lines.append(f"{k:>4}  {self.recall_at_k.get(k, 0.0):>9.1%}")
        lines.append("")
        lines.append(f"MRR      {self.mrr:.3f}")
        lines.append(f"hit rate {self.hit_rate:.1%}")
        if self.misses:
            lines.append("")
            lines.append(f"Trefferlose Fragen: {len(self.misses)}")
            for case in self.misses[:show_misses]:
                lines.append(f"  - {case.question}")
                lines.append(f"      erwartet: {', '.join(case.relevant[:3])}")
                lines.append(f"      bekommen: {', '.join(case.retrieved[:3])}")
        return "\n".join(lines)


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        cases.append(EvalCase(question=row["question"], relevant=list(row["relevant"]), note=row.get("note", "")))
    return cases


def _is_match(retrieved_ref: str, relevant: Iterable[str]) -> bool:
    doc_id, _, chunk_index = retrieved_ref.partition("#")
    for gold in relevant:
        gold_doc, _, gold_chunk = gold.partition("#")
        if gold_doc != doc_id:
            continue
        if not gold_chunk or not chunk_index or gold_chunk == chunk_index:
            return True
    return False


def score_case(case: EvalCase, hits: Sequence[SearchHit], ks: Sequence[int]) -> CaseResult:
    retrieved = [h.chunk.ref for h in hits]
    recall = {}
    max_k = max(ks) if ks else len(retrieved)
    for k in ks:
        window = retrieved[:k]
        found = {gold for gold in case.relevant for ref in window if _is_match(ref, [gold])}
        recall[k] = len(found) / len(case.relevant) if case.relevant else 0.0

    rr = 0.0
    for position, ref in enumerate(retrieved[: max(1, max_k)], start=1):
        if _is_match(ref, case.relevant):
            rr = 1.0 / position
            break
    return CaseResult(
        question=case.question,
        relevant=list(case.relevant),
        retrieved=retrieved,
        recall_at_k=recall,
        reciprocal_rank=rr,
        hit=rr > 0,
    )


def aggregate(results: list[CaseResult], ks: Sequence[int]) -> EvalReport:
    n = len(results) or 1
    recall = {k: sum(r.recall_at_k.get(k, 0.0) for r in results) / n for k in ks}
    mrr = sum(r.reciprocal_rank for r in results) / n
    hit_rate = sum(1 for r in results if r.hit) / n
    return EvalReport(
        ks=list(ks),
        n=len(results),
        recall_at_k=recall,
        mrr=mrr,
        hit_rate=hit_rate,
        misses=[r for r in results if not r.hit],
    )
