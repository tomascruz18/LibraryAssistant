"""Generate and incrementally persist paper embeddings in the SQLite catalog."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from storage import DEFAULT_DATABASE_PATH, connect_database, load_papers


DEFAULT_MODEL = "BAAI/bge-m3"
EMBEDDING_INPUT_VERSION = 1


@dataclass(frozen=True)
class EmbeddingSyncResult:
    """Counts produced by one incremental embedding synchronization."""

    total_papers: int
    generated: int
    reused: int
    skipped: int


def load_embedding_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def paper_text(paper: dict[str, Any]) -> str:
    """Create the versioned metadata representation used for paper search."""
    authors = paper.get("authors", [])
    authors_text = "; ".join(str(author).strip() for author in authors if str(author).strip())
    date = str(paper.get("date") or "").strip()
    abstract = str(paper.get("abstract") or "").strip()
    fields = (
        ("Authors", authors_text),
        ("Date", date),
        ("Abstract", abstract),
    )
    return "\n\n".join(f"{label}: {value}" for label, value in fields if value)


def embedding_input_hash(paper: dict[str, Any]) -> str:
    """Hash exactly the metadata that determines a paper embedding."""
    payload = {
        "version": EMBEDDING_INPUT_VERSION,
        "text": paper_text(paper),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


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


def sync_database_embeddings(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 16,
) -> EmbeddingSyncResult:
    """Embed only papers whose metadata input or model has changed.

    Vectors are stored in ``paper_embeddings`` and are keyed by Zotero's stable item
    key. Their input hash prevents re-embedding unchanged metadata; changing the
    embedding model deliberately refreshes every vector.
    """
    papers = [
        paper
        for paper in load_papers(database_path)
        if paper["is_supported"]
    ]
    eligible = [paper for paper in papers if paper_text(paper)]
    skipped = len(papers) - len(eligible)

    with closing(connect_database(database_path)) as connection:
        rows = connection.execute(
            "SELECT zotero_key, embedding_model, input_hash FROM paper_embeddings"
        ).fetchall()
    existing = {
        row["zotero_key"]: (row["embedding_model"], row["input_hash"])
        for row in rows
    }
    pending = [
        paper
        for paper in eligible
        if existing.get(paper["id"]) != (model_name, embedding_input_hash(paper))
    ]

    if pending:
        model = load_embedding_model(model_name)
        vectors = generate_embeddings(
            [paper_text(paper) for paper in pending], model, batch_size=batch_size
        )
        if vectors.ndim != 2 or len(vectors) != len(pending):
            raise ValueError("Embedding model returned an unexpected vector array.")
        vectors = np.asarray(vectors, dtype=np.float32)
        _upsert_database_embeddings(database_path, pending, model_name, vectors)

    _remove_ineligible_embeddings(database_path)
    return EmbeddingSyncResult(
        total_papers=len(papers),
        generated=len(pending),
        reused=len(eligible) - len(pending),
        skipped=skipped,
    )


def _upsert_database_embeddings(
    database_path: str | Path,
    papers: Sequence[dict[str, Any]],
    model_name: str,
    vectors: np.ndarray,
) -> None:
    """Store float32 vectors with their exact Zotero-key association."""
    with closing(connect_database(database_path)) as connection:
        with connection:
            for paper, vector in zip(papers, vectors, strict=True):
                connection.execute(
                    """
                    INSERT INTO paper_embeddings (
                        zotero_key, embedding_model, input_hash, dimensions, vector
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(zotero_key) DO UPDATE SET
                        embedding_model = excluded.embedding_model,
                        input_hash = excluded.input_hash,
                        dimensions = excluded.dimensions,
                        vector = excluded.vector,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        paper["id"],
                        model_name,
                        embedding_input_hash(paper),
                        int(vector.size),
                        vector.tobytes(),
                    ),
                )


def _remove_ineligible_embeddings(database_path: str | Path) -> None:
    """Discard vectors for papers removed from the searchable catalog."""
    with closing(connect_database(database_path)) as connection:
        with connection:
            connection.execute(
                """
                DELETE FROM paper_embeddings
                WHERE zotero_key IN (
                    SELECT zotero_key FROM papers
                    WHERE is_deleted = 1 OR is_supported = 0
                )
                """
            )


def load_database_embedding_index(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    model_name: str = DEFAULT_MODEL,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Load papers and vectors in one verified order for semantic search.

    The order is derived from the Zotero-key join, never from a separately maintained
    array index, so a vector cannot silently become associated with another paper.
    """
    stored_papers = {paper["id"]: paper for paper in load_papers(database_path)}
    with closing(connect_database(database_path)) as connection:
        rows = connection.execute(
            """
            SELECT zotero_key, dimensions, vector
            FROM paper_embeddings
            WHERE embedding_model = ?
            ORDER BY zotero_key
            """,
            (model_name,),
        ).fetchall()

    papers: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for row in rows:
        paper = stored_papers.get(row["zotero_key"])
        if paper is None or not paper["is_supported"]:
            continue
        vector = np.frombuffer(row["vector"], dtype=np.float32)
        if vector.size != row["dimensions"]:
            raise ValueError(f"Invalid stored embedding for Zotero key {paper['id']}.")
        papers.append(paper)
        vectors.append(vector.copy())

    if not vectors:
        return papers, np.empty((0, 0), dtype=np.float32)
    dimensions = vectors[0].size
    if any(vector.size != dimensions for vector in vectors):
        raise ValueError("Stored embeddings have inconsistent dimensions.")
    return papers, np.vstack(vectors)
