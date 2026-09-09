"""Optional second stage: ColBERT reranking with MaxSim.

Used when the first-stage retriever is not good enough. Retrieving more
candidates and reranking them is usually cheaper and more effective than
training an adapter on the generator.

    pip install pylate
"""

from __future__ import annotations

from .index import SearchHit


class ColBERTReranker:
    def __init__(self, model: str = "LiquidAI/LFM2.5-ColBERT-350M") -> None:
        try:
            from pylate import models  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "reranking needs PyLate: pip install pylate"
            ) from exc
        self.model = models.ColBERT(model_name_or_path=model, trust_remote_code=True)
        if self.model.tokenizer.pad_token is None:
            self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        self.name = model

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        query_vec = self.model.encode([query], is_query=True)
        doc_vecs = self.model.encode([h.chunk.text for h in hits], is_query=False)
        scored = [_maxsim(query_vec[0], doc) for doc in doc_vecs]
        order = sorted(range(len(hits)), key=lambda i: scored[i], reverse=True)[:top_k]
        return [
            SearchHit(chunk=hits[i].chunk, score=float(scored[i]), rank=rank)
            for rank, i in enumerate(order, start=1)
        ]


def _maxsim(query_vec, doc_vec) -> float:
    """ColBERT similarity: sum over query tokens of the best matching doc token."""
    import torch

    scores = query_vec @ doc_vec.T
    return float(torch.max(scores, dim=1).values.sum())
