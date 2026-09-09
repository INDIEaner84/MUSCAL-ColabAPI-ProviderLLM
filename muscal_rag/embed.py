"""Embedders.

Two backends:

``sentence-transformers``
    The real one. Loads ``LiquidAI/LFM2.5-Embedding-350M`` and encodes with the
    asymmetric ``prompt_name`` prefixes the model was trained with
    (``"query"`` / ``"document"``).

``hashing``
    Deterministic bag-of-words hashing into a fixed-width vector. It has no
    semantics -- it exists so the *pipeline* (chunking, indexing, metrics, CLI)
    is testable without downloading 700 MB or owning a GPU.

    Never read retrieval numbers produced by this backend as quality signal.
    The CLI refuses to print a quality verdict when it is active.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable

import numpy as np


class Embedder:
    name: str = "base"
    dim: int = 0

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    def __init__(
        self,
        model: str = "LiquidAI/LFM2.5-Embedding-350M",
        *,
        query_prompt: str = "query",
        document_prompt: str = "document",
        normalize: bool = True,
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # imported lazily

        self.model = SentenceTransformer(model, trust_remote_code=True)
        self.query_prompt = query_prompt
        self.document_prompt = document_prompt
        self.normalize = normalize
        self.batch_size = batch_size
        self.name = f"sentence-transformers:{model}"
        self.dim = int(self.model.get_sentence_embedding_dimension() or 0)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            list(texts),
            prompt_name=self.document_prompt,
            normalize_embeddings=self.normalize,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            list(texts),
            prompt_name=self.query_prompt,
            normalize_embeddings=self.normalize,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )


class HashingEmbedder(Embedder):
    """Pipeline smoke test only -- not a retrieval model."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.name = f"hashing:{dim}"

    def _encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
            bucket = int.from_bytes(digest, "big") % self.dim
            vec[bucket] += 1.0
        # Sublinear weighting so long documents do not dominate by length.
        np.log1p(vec, out=vec)
        norm = float(np.linalg.norm(vec))
        if norm:
            vec /= norm
        return vec

    @staticmethod
    def _tokens(text: str) -> Iterable[str]:
        return re.findall(r"[\wÄÖÜäöüß]+", text.lower())

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._encode(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode_documents(texts)


def build_embedder(cfg) -> Embedder:
    """cfg is a RagConfig; imported lazily to keep the module import-light."""
    emb = cfg.embedding
    if emb.backend == "hashing":
        return HashingEmbedder(dim=emb.dim)
    if emb.backend == "sentence-transformers":
        return SentenceTransformerEmbedder(
            emb.model,
            query_prompt=emb.query_prompt,
            document_prompt=emb.document_prompt,
            normalize=emb.normalize,
            batch_size=emb.batch_size,
        )
    raise ValueError(f"unknown embedding backend: {emb.backend!r}")


def cosine_scores(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """(n_queries, n_docs) similarity matrix. Vectors are expected normalised."""
    return np.asarray(query_vecs @ doc_vecs.T, dtype=np.float32)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def _unused_cosine(a: np.ndarray, b: np.ndarray) -> float:  # pragma: no cover
    return float(math.sqrt(max(0.0, 1.0 - float(np.dot(a, b)))))
