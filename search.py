"""Semantic, BM25, and hybrid paper search."""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


STOPWORDS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(word) >= 3 and word not in STOPWORDS
    ]


def build_bm25(texts: Sequence[str]) -> BM25Okapi:
    return BM25Okapi([tokenize(text) for text in texts])


def _normalize(scores: np.ndarray) -> np.ndarray:
    minimum = float(np.min(scores))
    maximum = float(np.max(scores))
    if maximum == minimum:
        return np.zeros_like(scores, dtype=float)
    return (scores - minimum) / (maximum - minimum)


def _results(
    order: np.ndarray,
    papers: Sequence[dict[str, Any]],
    semantic_scores: np.ndarray,
    lexical_scores: np.ndarray,
    final_scores: np.ndarray,
    k: int,
) -> list[dict[str, Any]]:
    results = []
    for rank, index in enumerate(order[:k], start=1):
        paper = papers[int(index)]
        results.append(
            {
                **paper,
                "rank": rank,
                "score": float(final_scores[index]),
                "semantic_score": float(semantic_scores[index]),
                "bm25_score": float(lexical_scores[index]),
            }
        )
    return results


def semantic_scores(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
) -> np.ndarray:
    query_embedding = model.encode([query], normalize_embeddings=True)
    return np.asarray(query_embedding @ embeddings.T)[0]


def semantic_search(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    papers: Sequence[dict[str, Any]],
    *,
    k: int = 10,
) -> list[dict[str, Any]]:
    scores = semantic_scores(query, model, embeddings)
    zeros = np.zeros_like(scores)
    return _results(np.argsort(scores)[::-1], papers, scores, zeros, scores, k)


def bm25_search(
    query: str,
    bm25: BM25Okapi,
    papers: Sequence[dict[str, Any]],
    *,
    k: int = 10,
) -> list[dict[str, Any]]:
    scores = np.asarray(bm25.get_scores(tokenize(query)))
    zeros = np.zeros_like(scores)
    return _results(np.argsort(scores)[::-1], papers, zeros, scores, scores, k)


def hybrid_search(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    bm25: BM25Okapi,
    papers: Sequence[dict[str, Any]],
    *,
    semantic_weight: float = 0.8,
    k: int = 10,
) -> list[dict[str, Any]]:
    if not 0 <= semantic_weight <= 1:
        raise ValueError("semantic_weight must be between 0 and 1")

    semantic = semantic_scores(query, model, embeddings)
    lexical = np.asarray(bm25.get_scores(tokenize(query)))
    combined = (
        semantic_weight * _normalize(semantic)
        + (1 - semantic_weight) * _normalize(lexical)
    )
    return _results(
        np.argsort(combined)[::-1],
        papers,
        semantic,
        lexical,
        combined,
        k,
    )
