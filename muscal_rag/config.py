"""Configuration for the RAG index + evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EmbeddingConf:
    backend: str = "sentence-transformers"  # sentence-transformers | hashing
    model: str = "LiquidAI/LFM2.5-Embedding-350M"
    # LFM2.5-Embedding is trained with asymmetric prefixes; both are required.
    query_prompt: str = "query"
    document_prompt: str = "document"
    normalize: bool = True
    batch_size: int = 32
    dim: int = 1024  # only used by the hashing backend


@dataclass
class ChunkingConf:
    chunk_size: int = 512  # tokens (the embedding model's document length)
    overlap: int = 64  # tokens
    chars_per_token: float = 4.0  # budget heuristic; no tokenizer needed


@dataclass
class RetrievalConf:
    top_k: int = 5
    rerank_model: str | None = None  # e.g. LiquidAI/LFM2.5-ColBERT-350M
    rerank_top_k: int = 20


@dataclass
class GenerationConf:
    enabled: bool = False
    model: str = "LiquidAI/LFM2.5-1.2B-Instruct"
    max_new_tokens: int = 256
    # The grounded prompt is the thing an adapter would later enforce.
    system_prompt: str = (
        "Beantworte die Frage ausschliesslich mit den gegebenen Ausschnitten. "
        "Nenne die Quellen-IDs. Wenn die Antwort nicht im Kontext steht, sage "
        "'Das steht nicht in den Unterlagen'."
    )


@dataclass
class EvalConf:
    eval_file: str = "data/rag_eval_example.jsonl"
    ks: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    show_misses: int = 5


@dataclass
class RagConfig:
    docs_dir: str = "data/rag_docs"
    index_dir: str = "outputs/rag/index"
    embedding: EmbeddingConf = field(default_factory=EmbeddingConf)
    chunking: ChunkingConf = field(default_factory=ChunkingConf)
    retrieval: RetrievalConf = field(default_factory=RetrievalConf)
    generation: GenerationConf = field(default_factory=GenerationConf)
    eval: EvalConf = field(default_factory=EvalConf)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RagConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RagConfig":
        valid = {f.name for f in fields(cls)}
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(f"unknown config section(s): {sorted(unknown)}")
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            # Scalars (docs_dir, index_dir) pass through; sections get built.
            kwargs[key] = _build(_SECTIONS[key], value) if key in _SECTIONS else value
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


_SECTIONS = {
    "embedding": EmbeddingConf,
    "chunking": ChunkingConf,
    "retrieval": RetrievalConf,
    "generation": GenerationConf,
    "eval": EvalConf,
}


def _build(cls, values: dict[str, Any] | None):
    if values is None:
        return cls()
    valid = {f.name for f in fields(cls)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"unknown key(s) for {cls.__name__}: {sorted(unknown)}")
    return cls(**values)
