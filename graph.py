"""Build, cluster, and persist the current paper-similarity graph."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import igraph as ig
import leidenalg
import numpy as np

from embeddings import DEFAULT_MODEL, load_database_embedding_index
from storage import DEFAULT_DATABASE_PATH, connect_database


DEFAULT_NEIGHBORS = 10
DEFAULT_SIMILARITY_PERCENTILE = 90.0


@dataclass(frozen=True)
class GraphSyncResult:
    """Counts and cache state from one graph-and-cluster synchronization."""

    rebuilt: bool
    papers: int
    edges: int
    clusters: int
    similarity_threshold: float | None


def build_similarity_graph(
    embeddings: np.ndarray,
    papers: Sequence[dict[str, Any]] | None = None,
    *,
    neighbors: int = DEFAULT_NEIGHBORS,
    percentile: float = DEFAULT_SIMILARITY_PERCENTILE,
) -> ig.Graph:
    """Build an undirected graph using global percentile and local k-NN rules."""
    if not 0 < percentile < 100:
        raise ValueError("percentile must be between 0 and 100.")
    if neighbors < 1:
        raise ValueError("neighbors must be at least 1.")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array.")
    if len(embeddings) < 2:
        graph = ig.Graph(n=len(embeddings), directed=False)
        if papers is not None:
            if len(papers) != len(embeddings):
                raise ValueError("papers and embeddings must have the same length.")
            graph.vs["zotero_key"] = [str(paper["id"]) for paper in papers]
            graph.vs["title"] = [str(paper.get("title", "")) for paper in papers]
        graph["similarity_threshold"] = None
        return graph

    similarity = embeddings @ embeddings.T
    upper_row, upper_column = np.triu_indices(len(embeddings), k=1)
    threshold = float(np.percentile(similarity[upper_row, upper_column], percentile))
    edge_weights: dict[tuple[int, int], float] = {}

    for source in range(len(embeddings)):
        candidates = np.argsort(similarity[source])[::-1]
        selected = 0
        for target in candidates:
            target = int(target)
            if source == target:
                continue
            weight = float(similarity[source, target])
            if weight < threshold:
                break
            edge = tuple(sorted((source, target)))
            edge_weights[edge] = max(weight, edge_weights.get(edge, -1.0))
            selected += 1
            if selected >= neighbors:
                break

    graph = ig.Graph(n=len(embeddings), edges=list(edge_weights), directed=False)
    graph.es["weight"] = list(edge_weights.values())
    graph["similarity_threshold"] = threshold
    graph["neighbors"] = neighbors
    graph["percentile"] = percentile
    if papers is not None:
        if len(papers) != len(embeddings):
            raise ValueError("papers and embeddings must have the same length.")
        graph.vs["zotero_key"] = [str(paper["id"]) for paper in papers]
        graph.vs["title"] = [str(paper.get("title", "")) for paper in papers]
    return graph


def detect_communities(graph: ig.Graph) -> list[int]:
    """Return a Leiden community number for each graph vertex."""
    if graph.ecount() == 0:
        return list(range(graph.vcount()))
    partition = leidenalg.find_partition(
        graph,
        leidenalg.ModularityVertexPartition,
        weights="weight",
    )
    return partition.membership


def sync_database_graph(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    model_name: str = DEFAULT_MODEL,
    neighbors: int = DEFAULT_NEIGHBORS,
    percentile: float = DEFAULT_SIMILARITY_PERCENTILE,
) -> GraphSyncResult:
    """Build and save a graph only when the persisted embeddings/configuration changed."""
    papers, embeddings = load_database_embedding_index(
        database_path, model_name=model_name
    )
    fingerprint = database_embedding_fingerprint(database_path, model_name)
    state = _load_graph_state(database_path, model_name)
    if state and (
        state["input_fingerprint"] == fingerprint
        and state["neighbors"] == neighbors
        and state["similarity_percentile"] == percentile
    ):
        return GraphSyncResult(
            rebuilt=False,
            papers=len(papers),
            edges=_count_rows(database_path, "paper_similarity_edges", model_name),
            clusters=_count_distinct_clusters(database_path, model_name),
            similarity_threshold=state["similarity_threshold"],
        )

    graph = build_similarity_graph(
        embeddings, papers, neighbors=neighbors, percentile=percentile
    )
    memberships = detect_communities(graph)
    _save_graph(
        database_path,
        model_name=model_name,
        input_fingerprint=fingerprint,
        neighbors=neighbors,
        percentile=percentile,
        graph=graph,
        memberships=memberships,
    )
    return GraphSyncResult(
        rebuilt=True,
        papers=len(papers),
        edges=graph.ecount(),
        clusters=len(set(memberships)),
        similarity_threshold=graph["similarity_threshold"],
    )


def database_embedding_fingerprint(database_path: str | Path, model_name: str) -> str:
    """Fingerprint the current stored vector inputs for one embedding model."""
    with closing(connect_database(database_path)) as connection:
        rows = connection.execute(
            """
            SELECT zotero_key, input_hash
            FROM paper_embeddings
            WHERE embedding_model = ?
            ORDER BY zotero_key
            """,
            (model_name,),
        ).fetchall()
    payload = [(row["zotero_key"], row["input_hash"]) for row in rows]
    return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()


def _load_graph_state(
    database_path: str | Path, model_name: str
) -> Any | None:
    with closing(connect_database(database_path)) as connection:
        return connection.execute(
            "SELECT * FROM similarity_graph_state WHERE embedding_model = ?",
            (model_name,),
        ).fetchone()


def _count_rows(database_path: str | Path, table: str, model_name: str) -> int:
    with closing(connect_database(database_path)) as connection:
        return int(
            connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE embedding_model = ?", (model_name,)
            ).fetchone()[0]
        )


def _count_distinct_clusters(database_path: str | Path, model_name: str) -> int:
    with closing(connect_database(database_path)) as connection:
        return int(
            connection.execute(
                """
                SELECT COUNT(DISTINCT cluster_id) FROM paper_clusters
                WHERE embedding_model = ?
                """,
                (model_name,),
            ).fetchone()[0]
        )


def _save_graph(
    database_path: str | Path,
    *,
    model_name: str,
    input_fingerprint: str,
    neighbors: int,
    percentile: float,
    graph: ig.Graph,
    memberships: Sequence[int],
) -> None:
    """Replace the current model graph and cluster memberships atomically."""
    now = datetime.now(timezone.utc).isoformat()
    with closing(connect_database(database_path)) as connection:
        with connection:
            connection.execute(
                "DELETE FROM paper_similarity_edges WHERE embedding_model = ?",
                (model_name,),
            )
            connection.execute(
                "DELETE FROM paper_clusters WHERE embedding_model = ?", (model_name,)
            )
            for edge in graph.es:
                source, target = edge.tuple
                source_key = graph.vs[source]["zotero_key"]
                target_key = graph.vs[target]["zotero_key"]
                connection.execute(
                    """
                    INSERT INTO paper_similarity_edges (
                        embedding_model, source_key, target_key, similarity
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        model_name,
                        min(source_key, target_key),
                        max(source_key, target_key),
                        float(edge["weight"]),
                    ),
                )
            for vertex, cluster_id in enumerate(memberships):
                connection.execute(
                    """
                    INSERT INTO paper_clusters (embedding_model, zotero_key, cluster_id)
                    VALUES (?, ?, ?)
                    """,
                    (model_name, graph.vs[vertex]["zotero_key"], int(cluster_id)),
                )
            connection.execute(
                """
                INSERT INTO similarity_graph_state (
                    embedding_model, input_fingerprint, neighbors,
                    similarity_percentile, similarity_threshold, built_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(embedding_model) DO UPDATE SET
                    input_fingerprint = excluded.input_fingerprint,
                    neighbors = excluded.neighbors,
                    similarity_percentile = excluded.similarity_percentile,
                    similarity_threshold = excluded.similarity_threshold,
                    built_at = excluded.built_at
                """,
                (
                    model_name,
                    input_fingerprint,
                    neighbors,
                    percentile,
                    graph["similarity_threshold"],
                    now,
                ),
            )
