"""Build an index, search it, evaluate it -- the three verbs of the baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import documents as docmod
from .chunking import chunk_text
from .config import RagConfig
from .embed import build_embedder
from .index import NumpyIndex
from .metrics import EvalCase, aggregate, score_case


def build_index(cfg: RagConfig) -> NumpyIndex:
    docs = docmod.load_documents(cfg.docs_dir)
    if not docs:
        raise SystemExit(f"no documents found in {cfg.docs_dir}")

    chunks = []
    for doc in docs:
        chunks.extend(
            chunk_text(
                doc.text,
                doc.id,
                chunk_size=cfg.chunking.chunk_size,
                overlap=cfg.chunking.overlap,
                chars_per_token=cfg.chunking.chars_per_token,
            )
        )
    print(f"[docs] {len(docs)} Dokumente -> {len(chunks)} Chunks "
          f"(chunk_size={cfg.chunking.chunk_size}, overlap={cfg.chunking.overlap})")

    embedder = build_embedder(cfg)
    print(f"[embed] {embedder.name}")
    vectors = np.asarray(embedder.encode_documents([c.text for c in chunks]), dtype=np.float32)

    index = NumpyIndex(
        vectors,
        chunks,
        meta={
            "embedder": embedder.name,
            "dim": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
            "chunk_size": cfg.chunking.chunk_size,
            "overlap": cfg.chunking.overlap,
            "documents": len(docs),
            "chunks": len(chunks),
        },
    )
    index.save(cfg.index_dir)
    print(f"[write] index -> {cfg.index_dir}")
    return index


def search(cfg: RagConfig, questions: list[str], *, index: NumpyIndex | None = None) -> list[list]:
    index = index or NumpyIndex.load(cfg.index_dir)
    embedder = build_embedder(cfg)
    vectors = np.asarray(embedder.encode_queries(questions), dtype=np.float32)
    return index.search(vectors, top_k=cfg.retrieval.top_k)


def evaluate(cfg: RagConfig, cases: list[EvalCase], *, show_misses: int | None = None):
    index = NumpyIndex.load(cfg.index_dir)
    embedder = build_embedder(cfg)
    query_vecs = np.asarray(embedder.encode_queries([c.question for c in cases]), dtype=np.float32)
    # Retrieve deeper than top_k so every k in the eval list is measurable.
    depth = max(cfg.eval.ks)
    hits = index.search(query_vecs, top_k=depth)

    if cfg.retrieval.rerank_model:
        from .rerank import ColBERTReranker

        reranker = ColBERTReranker(cfg.retrieval.rerank_model)
        print(f"[rerank] {reranker.name} (top {cfg.retrieval.rerank_top_k} -> {depth})")
        deeper = index.search(query_vecs, top_k=cfg.retrieval.rerank_top_k)
        hits = [
            reranker.rerank(case.question, h, depth)
            for case, h in zip(cases, deeper)
        ]

    if depth >= len(index):
        print(f"[warn] k={depth} bei nur {len(index)} Chunks -- recall@{depth} ist dann "
              f"aussagelos. Korpus vergroessern oder ks verkleinern.")

    results = [score_case(case, h, cfg.eval.ks) for case, h in zip(cases, hits)]
    report = aggregate(results, cfg.eval.ks)
    report.misses = [r for r in results if not r.hit]
    return report, results


def grounded_prompt(cfg: RagConfig, question: str, hits) -> str:
    """The context block a generator would see. Also what an adapter must learn to obey."""
    blocks = []
    for hit in hits:
        blocks.append(f"[{hit.chunk.ref}] {hit.chunk.text}")
    context = "\n\n".join(blocks)
    return (
        f"{cfg.generation.system_prompt}\n\n"
        f"# Kontext\n{context}\n\n"
        f"# Frage\n{question}\n\n"
        f"# Antwort"
    )


def generate_answer(cfg: RagConfig, question: str, hits) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.generation.model)
    model = AutoModelForCausalLM.from_pretrained(cfg.generation.model, dtype="auto", device_map="auto")
    prompt = grounded_prompt(cfg, question, hits)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=cfg.generation.max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def default_index_path(cfg: RagConfig) -> Path:
    return Path(cfg.index_dir)
