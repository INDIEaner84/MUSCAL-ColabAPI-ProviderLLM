"""MUSCAL RAG baseline: chunk, embed, retrieve, measure.

Built to answer one question without spending GPU money first: is retrieval
good enough that no adapter is needed at all?
"""

from .config import RagConfig
from .index import NumpyIndex
from .metrics import EvalCase, aggregate, load_eval_cases, score_case

__all__ = [
    "RagConfig",
    "NumpyIndex",
    "EvalCase",
    "aggregate",
    "load_eval_cases",
    "score_case",
]
