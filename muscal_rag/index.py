"""A deliberately boring vector index: numpy + cosine, saved as npz + json.

For the corpus sizes that matter here (hundreds to low hundreds of thousands of
chunks) this is faster to set up and easier to debug than a vector database, and
it keeps the whole baseline reproducible from a single command.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .embed import cosine_scores


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    rank: int


class NumpyIndex:
    def __init__(self, vectors: np.ndarray, chunks: list[Chunk], meta: dict | None = None) -> None:
        self.vectors = vectors.astype(np.float32)
        self.chunks = chunks
        self.meta = meta or {}

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query_vecs: np.ndarray, top_k: int = 5) -> list[list[SearchHit]]:
        scores = cosine_scores(query_vecs, self.vectors)
        results: list[list[SearchHit]] = []
        for row in scores:
            k = min(top_k, len(row))
            top = np.argpartition(-row, k - 1)[:k] if k else np.array([], dtype=int)
            top = top[np.argsort(-row[top])]
            results.append(
                [SearchHit(chunk=self.chunks[i], score=float(row[i]), rank=r) for r, i in enumerate(top, start=1)]
            )
        return results

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "vectors.npz", vectors=self.vectors)
        (out_dir / "chunks.json").write_text(
            json.dumps(
                [{"id": c.id, "doc_id": c.doc_id, "index": c.index, "text": c.text} for c in self.chunks],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (out_dir / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: str | Path) -> "NumpyIndex":
        index_dir = Path(index_dir)
        vectors = np.load(index_dir / "vectors.npz")["vectors"]
        chunks = [
            Chunk(id=row["id"], doc_id=row["doc_id"], index=row["index"], text=row["text"])
            for row in json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        ]
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8")) if (index_dir / "meta.json").exists() else {}
        return cls(vectors, chunks, meta)
