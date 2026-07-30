"""Generate and persist concise names for the current Leiden clusters."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from embeddings import DEFAULT_MODEL as DEFAULT_EMBEDDING_MODEL
from llm import (
    CLUSTER_LABEL_PIPELINE_VERSION,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    DEFAULT_MODEL as DEFAULT_LLM_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    generate_cluster_label,
)
from storage import DEFAULT_DATABASE_PATH, connect_database


DEFAULT_REPRESENTATIVE_PAPERS = 5


@dataclass(frozen=True)
class ClusterLabelSyncResult:
    generated: int
    reused: int
    title_based: int


def label_database_clusters(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    llm_model: str = DEFAULT_LLM_MODEL,
    representative_papers: int = DEFAULT_REPRESENTATIVE_PAPERS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
    refresh: bool = False,
) -> ClusterLabelSyncResult:
    """Label current clusters, reusing labels tied to the current graph fingerprint."""
    if representative_papers < 1:
        raise ValueError("representative_papers must be at least 1.")
    graph_fingerprint = current_graph_fingerprint(database_path, embedding_model)
    representatives = _cluster_representatives(
        database_path, embedding_model, representative_papers
    )
    existing = _current_labels(database_path, embedding_model, graph_fingerprint)
    generated = reused = title_based = 0

    for cluster_id, papers in representatives.items():
        current = existing.get(cluster_id)
        if current and current["source"] == "manual":
            reused += 1
            continue
        if current and not refresh and current["source"] == "title":
            reused += 1
            continue
        if (
            current
            and not refresh
            and current["source"] == "llm"
            and current["llm_model"] == llm_model
            and current["label_pipeline_version"] == CLUSTER_LABEL_PIPELINE_VERSION
        ):
            reused += 1
            continue
        if len(papers) == 1:
            label = _title_label(str(papers[0]["title"]))
            description = f"Single-paper cluster represented by: {papers[0]['title']}"
            source = "title"
            title_based += 1
        else:
            generated_label = generate_cluster_label(
                papers,
                model=llm_model,
                timeout_seconds=timeout_seconds,
                context_window_tokens=context_window_tokens,
            )
            label = generated_label["label"]
            description = generated_label["description"]
            source = "llm"
            generated += 1
        _save_label(
            database_path,
            embedding_model=embedding_model,
            graph_fingerprint=graph_fingerprint,
            cluster_id=cluster_id,
            label=label,
            description=description,
            source=source,
            llm_model=llm_model if source == "llm" else None,
        )
    return ClusterLabelSyncResult(generated, reused, title_based)


def current_graph_fingerprint(database_path: str | Path, embedding_model: str) -> str:
    with closing(connect_database(database_path)) as connection:
        row = connection.execute(
            """
            SELECT input_fingerprint, neighbors, similarity_percentile
            FROM similarity_graph_state
            WHERE embedding_model = ?
            """,
            (embedding_model,),
        ).fetchone()
    if row is None:
        raise ValueError("No saved graph found. Run scripts\\build_graph.py first.")
    # Embedding changes and graph-parameter changes can both alter the meaning of a
    # Leiden cluster. Include both in the label identity.
    payload = {
        "embedding_input": row["input_fingerprint"],
        "neighbors": row["neighbors"],
        "similarity_percentile": row["similarity_percentile"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _cluster_representatives(
    database_path: str | Path,
    embedding_model: str,
    limit: int,
) -> dict[int, list[dict[str, Any]]]:
    """Pick papers with the greatest weighted graph degree in each cluster."""
    with closing(connect_database(database_path)) as connection:
        rows = connection.execute(
            """
            SELECT cluster.cluster_id, paper.zotero_key, paper.title, paper.abstract,
                   COALESCE(SUM(edge.similarity), 0) AS weighted_degree
            FROM paper_clusters AS cluster
            JOIN papers AS paper ON paper.zotero_key = cluster.zotero_key
            LEFT JOIN paper_similarity_edges AS edge
                ON edge.embedding_model = cluster.embedding_model
                AND (edge.source_key = paper.zotero_key OR edge.target_key = paper.zotero_key)
            WHERE cluster.embedding_model = ?
            GROUP BY cluster.cluster_id, paper.zotero_key, paper.title, paper.abstract
            ORDER BY cluster.cluster_id, weighted_degree DESC, paper.title COLLATE NOCASE
            """,
            (embedding_model,),
        ).fetchall()
    output: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        cluster_id = int(row["cluster_id"])
        selected = output.setdefault(cluster_id, [])
        if len(selected) < limit:
            selected.append(
                {"title": row["title"], "abstract": row["abstract"]}
            )
    return output


def _current_labels(
    database_path: str | Path, embedding_model: str, graph_fingerprint: str
) -> dict[int, Any]:
    with closing(connect_database(database_path)) as connection:
        rows = connection.execute(
            """
            SELECT cluster_id, source, llm_model, label_pipeline_version
            FROM cluster_labels
            WHERE embedding_model = ? AND graph_fingerprint = ?
            """,
            (embedding_model, graph_fingerprint),
        ).fetchall()
    return {int(row["cluster_id"]): row for row in rows}


def _title_label(title: str) -> str:
    """Use a short, readable title fragment for a one-paper community."""
    compact = " ".join(title.split())
    for separator in (":", "—", "–", "-"):
        if separator in compact:
            compact = compact.split(separator, maxsplit=1)[0].strip()
            break
    return compact[:100] or "Untitled paper"


def _save_label(
    database_path: str | Path,
    *,
    embedding_model: str,
    graph_fingerprint: str,
    cluster_id: int,
    label: str,
    description: str,
    source: str,
    llm_model: str | None,
) -> None:
    with closing(connect_database(database_path)) as connection:
        with connection:
            connection.execute(
                """
                INSERT INTO cluster_labels (
                    embedding_model, graph_fingerprint, cluster_id, label, description,
                    source, llm_model, label_pipeline_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(embedding_model, graph_fingerprint, cluster_id) DO UPDATE SET
                    label = excluded.label,
                    description = excluded.description,
                    source = excluded.source,
                    llm_model = excluded.llm_model,
                    label_pipeline_version = excluded.label_pipeline_version,
                    updated_at = excluded.updated_at
                """,
                (
                    embedding_model,
                    graph_fingerprint,
                    cluster_id,
                    label,
                    description,
                    source,
                    llm_model,
                    CLUSTER_LABEL_PIPELINE_VERSION,
                    datetime.now(timezone.utc).isoformat(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
