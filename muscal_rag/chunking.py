"""Chunking: split documents into embedding-sized pieces.

Budget is expressed in characters, derived from a token estimate
(``chars_per_token``). That avoids pulling in a tokenizer just to chunk, which
matters because the chunker runs on CPU for every corpus.

Paragraphs are the unit: a chunk never starts mid-paragraph unless a single
paragraph exceeds the budget. Overlap carries the tail of the previous chunk so
a sentence spanning a boundary is not lost.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    doc_id: str
    index: int
    text: str

    @property
    def ref(self) -> str:
        """Stable reference used in eval files: ``doc_id#3``."""
        return f"{self.doc_id}#{self.index}"


def _split_paragraphs(text: str) -> list[str]:
    parts = []
    for block in text.split("\n\n"):
        block = block.strip()
        if block:
            parts.append(block)
    return parts or ([text.strip()] if text.strip() else [])


def chunk_text(text: str, doc_id: str, *, chunk_size: int, overlap: int, chars_per_token: float = 4.0) -> list[Chunk]:
    budget = max(200, int(chunk_size * chars_per_token))
    overlap_chars = max(0, int(overlap * chars_per_token))

    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        body = "\n\n".join(current).strip()
        if body:
            chunks.append(Chunk(id="", doc_id=doc_id, index=len(chunks), text=body))
        current, current_len = [], 0

    for paragraph in _split_paragraphs(text):
        # A single oversized paragraph is hard-split instead of dropped.
        while len(paragraph) > budget:
            flush()
            head, paragraph = paragraph[:budget], paragraph[budget - overlap_chars :]
            chunks.append(Chunk(id="", doc_id=doc_id, index=len(chunks), text=head.strip()))
        if current_len + len(paragraph) + 2 > budget and current:
            flush()
            if overlap_chars and chunks:
                tail = chunks[-1].text[-overlap_chars:]
                current, current_len = [tail], len(tail)
        current.append(paragraph)
        current_len += len(paragraph) + 2

    flush()

    for chunk in chunks:
        chunk.id = chunk.ref
    return chunks
