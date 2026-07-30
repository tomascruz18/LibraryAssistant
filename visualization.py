"""Persist UMAP paper coordinates and render the stored similarity graph."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go
import umap
import igraph as ig

from embeddings import DEFAULT_MODEL, load_database_embedding_index
from graph import database_embedding_fingerprint
from storage import DEFAULT_DATABASE_PATH, connect_database


DEFAULT_UMAP_NEIGHBORS = 15
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_UMAP_RANDOM_STATE = 42


@dataclass(frozen=True)
class ProjectionSyncResult:
    rebuilt: bool
    papers: int


def project_embeddings(
    embeddings: np.ndarray,
    *,
    neighbors: int = DEFAULT_UMAP_NEIGHBORS,
    min_dist: float = DEFAULT_UMAP_MIN_DIST,
    random_state: int = DEFAULT_UMAP_RANDOM_STATE,
) -> np.ndarray:
    """Project normalized paper vectors to two stable UMAP coordinates."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array.")
    if len(embeddings) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(embeddings) == 1:
        return np.zeros((1, 2), dtype=np.float32)
    effective_neighbors = min(neighbors, len(embeddings) - 1)
    return np.asarray(
        umap.UMAP(
            n_components=2,
            n_neighbors=effective_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=random_state,
        ).fit_transform(embeddings),
        dtype=np.float32,
    )


def sync_database_projection(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    model_name: str = DEFAULT_MODEL,
    neighbors: int = DEFAULT_UMAP_NEIGHBORS,
    min_dist: float = DEFAULT_UMAP_MIN_DIST,
    random_state: int = DEFAULT_UMAP_RANDOM_STATE,
) -> ProjectionSyncResult:
    """Compute UMAP only when its embedding inputs or settings changed."""
    papers, embeddings = load_database_embedding_index(
        database_path, model_name=model_name
    )
    fingerprint = database_embedding_fingerprint(database_path, model_name)
    with closing(connect_database(database_path)) as connection:
        state = connection.execute(
            "SELECT * FROM projection_state WHERE embedding_model = ?", (model_name,)
        ).fetchone()
    if state and (
        state["input_fingerprint"] == fingerprint
        and state["neighbors"] == neighbors
        and state["min_dist"] == min_dist
        and state["random_state"] == random_state
    ):
        return ProjectionSyncResult(rebuilt=False, papers=len(papers))

    coordinates = project_embeddings(
        embeddings,
        neighbors=neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    now = datetime.now(timezone.utc).isoformat()
    with closing(connect_database(database_path)) as connection:
        with connection:
            connection.execute(
                "DELETE FROM paper_projections WHERE embedding_model = ?", (model_name,)
            )
            connection.executemany(
                """
                INSERT INTO paper_projections (embedding_model, zotero_key, x, y)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (model_name, paper["id"], float(point[0]), float(point[1]))
                    for paper, point in zip(papers, coordinates, strict=True)
                ],
            )
            connection.execute(
                """
                INSERT INTO projection_state (
                    embedding_model, input_fingerprint, neighbors, min_dist,
                    random_state, built_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(embedding_model) DO UPDATE SET
                    input_fingerprint = excluded.input_fingerprint,
                    neighbors = excluded.neighbors,
                    min_dist = excluded.min_dist,
                    random_state = excluded.random_state,
                    built_at = excluded.built_at
                """,
                (model_name, fingerprint, neighbors, min_dist, random_state, now),
            )
    return ProjectionSyncResult(rebuilt=True, papers=len(papers))


def load_database_visualization(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    model_name: str = DEFAULT_MODEL,
) -> tuple[list[dict[str, Any]], np.ndarray, list[int], list[tuple[int, int, float]]]:
    """Load coordinates, communities, and graph edges in one Zotero-key order."""
    papers, _embeddings = load_database_embedding_index(database_path, model_name=model_name)
    papers_by_key = {paper["id"]: paper for paper in papers}
    with closing(connect_database(database_path)) as connection:
        projection_rows = connection.execute(
            """
            SELECT zotero_key, x, y FROM paper_projections
            WHERE embedding_model = ? ORDER BY zotero_key
            """,
            (model_name,),
        ).fetchall()
        cluster_rows = connection.execute(
            """
            SELECT zotero_key, cluster_id FROM paper_clusters
            WHERE embedding_model = ?
            """,
            (model_name,),
        ).fetchall()
        edge_rows = connection.execute(
            """
            SELECT source_key, target_key, similarity FROM paper_similarity_edges
            WHERE embedding_model = ?
            """,
            (model_name,),
        ).fetchall()

    cluster_by_key = {row["zotero_key"]: int(row["cluster_id"]) for row in cluster_rows}
    ordered_rows = [row for row in projection_rows if row["zotero_key"] in papers_by_key]
    ordered_papers = [papers_by_key[row["zotero_key"]] for row in ordered_rows]
    coordinates = np.asarray([[row["x"], row["y"]] for row in ordered_rows], dtype=np.float32)
    if not len(coordinates):
        coordinates = np.empty((0, 2), dtype=np.float32)
    communities = [cluster_by_key.get(paper["id"], -1) for paper in ordered_papers]
    index_by_key = {paper["id"]: index for index, paper in enumerate(ordered_papers)}
    edges = [
        (
            index_by_key[row["source_key"]],
            index_by_key[row["target_key"]],
            float(row["similarity"]),
        )
        for row in edge_rows
        if row["source_key"] in index_by_key and row["target_key"] in index_by_key
    ]
    return ordered_papers, coordinates, communities, edges


def plot_papers(
    coordinates: np.ndarray,
    papers: Sequence[dict[str, Any]],
    communities: Sequence[int],
    edges: Sequence[tuple[int, int, float]] = (),
) -> go.Figure:
    """Create the interactive UMAP scatter/graph view from persisted data."""
    return _plot_graph(coordinates, papers, communities, edges, "Paper similarity map (UMAP)")


def plot_force_graph(
    papers: Sequence[dict[str, Any]],
    communities: Sequence[int],
    edges: Sequence[tuple[int, int, float]],
) -> go.Figure:
    """Render the persisted similarity graph with a deterministic force layout."""
    graph = ig.Graph(n=len(papers), edges=[(source, target) for source, target, _ in edges])
    graph.es["weight"] = [weight for _, _, weight in edges]
    if len(papers) == 0:
        coordinates = np.empty((0, 2), dtype=np.float32)
    elif len(papers) == 1:
        coordinates = np.zeros((1, 2), dtype=np.float32)
    else:
        angles = np.linspace(0, 2 * np.pi, len(papers), endpoint=False)
        seed = np.column_stack((np.cos(angles), np.sin(angles))).tolist()
        coordinates = np.asarray(
            graph.layout_fruchterman_reingold(
                weights="weight", niter=500, seed=seed, grid="auto"
            ),
            dtype=np.float32,
        )
    return _plot_graph(
        coordinates, papers, communities, edges, "Paper similarity graph (force layout)"
    )


def _plot_graph(
    coordinates: np.ndarray,
    papers: Sequence[dict[str, Any]],
    communities: Sequence[int],
    edges: Sequence[tuple[int, int, float]],
    title: str,
) -> go.Figure:
    """Render common Plotly styling for either the UMAP or force layout."""
    if len(coordinates) != len(papers) or len(communities) != len(papers):
        raise ValueError("Coordinates, papers, and communities must have the same length.")
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for source, target, _weight in edges:
        edge_x.extend([coordinates[source, 0], coordinates[target, 0], None])
        edge_y.extend([coordinates[source, 1], coordinates[target, 1], None])

    hover = [str(paper.get("title", "Untitled")) for paper in papers]
    unique_clusters = sorted(set(communities))
    cluster_colors = {
        cluster: f"hsl({index * 360 / max(len(unique_clusters), 1):.0f}, 65%, 76%)"
        for index, cluster in enumerate(unique_clusters)
    }
    figure = go.Figure(
        data=[
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"width": 0.5, "color": "lightgray"},
                hoverinfo="none",
                showlegend=False,
            ),
            go.Scatter(
                x=coordinates[:, 0],
                y=coordinates[:, 1],
                mode="markers",
                customdata=hover,
                hovertemplate="<b>%{customdata}</b><extra></extra>",
                marker={
                    "size": 9,
                    "color": [cluster_colors[cluster] for cluster in communities],
                    "line": {"width": 0.5, "color": "#555555"},
                },
                showlegend=False,
            ),
        ]
    )
    figure.update_layout(
        title=title,
        hovermode="closest",
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure


def write_database_visualization(
    output_path: str | Path,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    model_name: str = DEFAULT_MODEL,
) -> Path:
    """Write the stored UMAP graph visualization as one self-contained HTML file."""
    papers, coordinates, communities, edges = load_database_visualization(
        database_path, model_name=model_name
    )
    if not papers:
        raise ValueError("No projected papers found. Build embeddings, graph, and UMAP first.")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plot_papers(coordinates, papers, communities, edges).write_html(target)
    return target


def write_database_force_visualization(
    output_path: str | Path,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    model_name: str = DEFAULT_MODEL,
) -> Path:
    """Write the stored graph as an interactive force-layout HTML file."""
    papers, _coordinates, communities, edges = load_database_visualization(
        database_path, model_name=model_name
    )
    if not papers:
        raise ValueError("No projected papers found. Build embeddings, graph, and UMAP first.")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plot_force_graph(papers, communities, edges).write_html(target)
    return target
