"""Generate and persist embeddings for paper text."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "BAAI/bge-m3"


def load_embedding_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def paper_text(paper: dict[str, Any]) -> str:
    """Create the text representation used for paper-level MVP search."""
    return f"{paper.get('title', '')}\n\n{paper.get('abstract', '')}".strip()


def generate_embeddings(
    texts: Sequence[str],
    model: SentenceTransformer,
    *,
    batch_size: int = 16,
) -> np.ndarray:
    return np.asarray(
        model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    )


def save_embeddings(path: str | Path, embeddings: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, embeddings)


def load_embeddings(path: str | Path) -> np.ndarray:
    return np.load(Path(path))
