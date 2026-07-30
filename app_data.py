"""Database-backed state and figures for the local Dash application."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go
from pyzotero import zotero

from clusters import current_graph_fingerprint
from embeddings import (
    DEFAULT_MODEL,
    load_database_embedding_index,
    load_embedding_model,
    paper_text,
)
from search import bm25_search, build_bm25, hybrid_search, semantic_search
from storage import DEFAULT_DATABASE_PATH, connect_database
from visualization import force_layout_coordinates, load_database_visualization


@lru_cache(maxsize=4)
def _cached_embedding_model(model_name: str) -> Any:
    """Load a query model once per app process."""
    return load_embedding_model(model_name)


def _pdf_in_attachment_directory(
    storage_root: Path, attachment_key: str
) -> Path | None:
    directory = storage_root / attachment_key
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob("*.pdf"))
    return candidates[0] if candidates else None


@lru_cache(maxsize=2_048)
def _resolve_pdf_path(
    zotero_key: str,
    stored_attachment_key: str,
    storage_root_text: str,
) -> Path | None:
    """Resolve saved or live-Zotero attachment identity to one local PDF."""
    storage_root = Path(storage_root_text)
    if stored_attachment_key:
        saved_path = _pdf_in_attachment_directory(
            storage_root, stored_attachment_key
        )
        if saved_path is not None:
            return saved_path
    try:
        client = zotero.Zotero(
            library_id="0", library_type="user", local=True
        )
        for child in client.children(zotero_key):
            data = child.get("data", {})
            filename = str(data.get("filename", "")).lower()
            if (
                data.get("contentType") == "application/pdf"
                or filename.endswith(".pdf")
            ):
                attachment_key = str(
                    child.get("key") or data.get("key") or ""
                )
                if attachment_key:
                    path = _pdf_in_attachment_directory(
                        storage_root, attachment_key
                    )
                    if path is not None:
                        return path
    except Exception:
        return None
    return None


@dataclass(frozen=True)
class ClusterInfo:
    cluster_id: int
    label: str
    description: str
    color: str
    paper_count: int


@dataclass
class LibraryAppData:
    """Immutable-at-runtime snapshot of the persisted library."""

    database_path: Path
    model_name: str
    papers: list[dict[str, Any]]
    embeddings: np.ndarray
    umap_coordinates: np.ndarray
    force_coordinates: np.ndarray
    communities: list[int]
    edges: list[tuple[int, int, float]]
    clusters: dict[int, ClusterInfo]
    bm25: Any

    @classmethod
    def load(
        cls,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
        *,
        model_name: str = DEFAULT_MODEL,
    ) -> "LibraryAppData":
        database = Path(database_path)
        if not database.is_file():
            raise FileNotFoundError(
                f"Database not found: {database}. Run scripts/build_all.py first."
            )
        papers, embeddings = load_database_embedding_index(
            database, model_name=model_name
        )
        (
            visualization_papers,
            umap_coordinates,
            communities,
            edges,
        ) = load_database_visualization(database, model_name=model_name)
        if not papers:
            raise ValueError(
                "No stored embeddings found. Run scripts/build_all.py first."
            )
        if [paper["id"] for paper in papers] != [
            paper["id"] for paper in visualization_papers
        ]:
            raise ValueError(
                "Stored projections do not match the embedding index. "
                "Run scripts/build_visualization.py."
            )

        cluster_rows = cls._load_cluster_rows(database, model_name)
        unique_clusters = sorted(set(communities))
        colors = {
            cluster_id: (
                f"hsl({index * 360 / max(len(unique_clusters), 1):.0f}, 65%, 76%)"
            )
            for index, cluster_id in enumerate(unique_clusters)
        }
        counts = {
            cluster_id: communities.count(cluster_id)
            for cluster_id in unique_clusters
        }
        clusters = {
            cluster_id: ClusterInfo(
                cluster_id=cluster_id,
                label=cluster_rows.get(cluster_id, {}).get(
                    "label", f"Cluster {cluster_id}"
                ),
                description=cluster_rows.get(cluster_id, {}).get("description", ""),
                color=colors[cluster_id],
                paper_count=counts[cluster_id],
            )
            for cluster_id in unique_clusters
        }
        return cls(
            database_path=database,
            model_name=model_name,
            papers=papers,
            embeddings=embeddings,
            umap_coordinates=umap_coordinates,
            force_coordinates=force_layout_coordinates(len(papers), edges),
            communities=communities,
            edges=edges,
            clusters=clusters,
            bm25=build_bm25([paper_text(paper) for paper in papers]),
        )

    @staticmethod
    def _load_cluster_rows(
        database_path: Path, model_name: str
    ) -> dict[int, dict[str, str]]:
        fingerprint = current_graph_fingerprint(database_path, model_name)
        with closing(connect_database(database_path)) as connection:
            rows = connection.execute(
                """
                SELECT cluster_id, label, description
                FROM cluster_labels
                WHERE embedding_model = ? AND graph_fingerprint = ?
                """,
                (model_name, fingerprint),
            ).fetchall()
        return {
            int(row["cluster_id"]): {
                "label": str(row["label"]),
                "description": str(row["description"]),
            }
            for row in rows
        }

    @property
    def papers_by_key(self) -> dict[str, dict[str, Any]]:
        return {paper["id"]: paper for paper in self.papers}

    @property
    def index_by_key(self) -> dict[str, int]:
        return {paper["id"]: index for index, paper in enumerate(self.papers)}

    def cluster_for_key(self, zotero_key: str) -> ClusterInfo | None:
        index = self.index_by_key.get(zotero_key)
        if index is None:
            return None
        return self.clusters.get(self.communities[index])

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        *,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []
        if mode == "bm25":
            return bm25_search(query, self.bm25, self.papers, k=limit)
        model = _cached_embedding_model(self.model_name)
        if mode == "semantic":
            return semantic_search(
                query, model, self.embeddings, self.papers, k=limit
            )
        return hybrid_search(
            query,
            model,
            self.embeddings,
            self.bm25,
            self.papers,
            k=limit,
        )

    def papers_in_clusters(
        self, cluster_ids: Sequence[int], *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        selected = {
            cluster_id for cluster_id in cluster_ids if cluster_id in self.clusters
        }
        papers = [
            paper
            for paper, cluster_id in zip(
                self.papers, self.communities, strict=True
            )
            if cluster_id in selected
        ]
        return papers if limit is None else papers[:limit]

    def figure(
        self,
        layout_mode: str,
        cluster_ids: Sequence[int] | None = None,
        selected_key: str | None = None,
    ) -> go.Figure:
        """Build a filtered graph with stable Zotero keys in click data."""
        coordinates = (
            self.force_coordinates
            if layout_mode == "force"
            else self.umap_coordinates
        )
        allowed_clusters = (
            set(cluster_ids) if cluster_ids else set(self.clusters)
        )
        visible_indices = [
            index
            for index, cluster_id in enumerate(self.communities)
            if cluster_id in allowed_clusters
        ]
        visible = set(visible_indices)
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        for source, target, _weight in self.edges:
            if source not in visible or target not in visible:
                continue
            edge_x.extend(
                [coordinates[source, 0], coordinates[target, 0], None]
            )
            edge_y.extend(
                [coordinates[source, 1], coordinates[target, 1], None]
            )

        node_x = [float(coordinates[index, 0]) for index in visible_indices]
        node_y = [float(coordinates[index, 1]) for index in visible_indices]
        custom_data = [
            [self.papers[index]["id"], self.papers[index].get("title", "Untitled")]
            for index in visible_indices
        ]
        node_colors = [
            self.clusters[self.communities[index]].color
            for index in visible_indices
        ]
        figure = go.Figure(
            data=[
                go.Scattergl(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line={"width": 0.7, "color": "rgba(122, 133, 150, 0.25)"},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                go.Scattergl(
                    x=node_x,
                    y=node_y,
                    mode="markers",
                    customdata=custom_data,
                    hovertemplate="<b>%{customdata[1]}</b><extra></extra>",
                    marker={
                        "size": 10,
                        "color": node_colors,
                        "line": {"width": 1, "color": "#5d6676"},
                    },
                    showlegend=False,
                ),
            ]
        )
        selected_index = self.index_by_key.get(selected_key or "")
        if selected_index is not None and selected_index in visible:
            figure.add_trace(
                go.Scattergl(
                    x=[float(coordinates[selected_index, 0])],
                    y=[float(coordinates[selected_index, 1])],
                    mode="markers",
                    marker={
                        "size": 19,
                        "color": "rgba(255,255,255,0)",
                        "line": {"width": 3, "color": "#182131"},
                    },
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        figure.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="closest",
            dragmode="pan",
            uirevision=f"{layout_mode}:{','.join(map(str, sorted(allowed_clusters)))}",
            xaxis={"visible": False, "fixedrange": False},
            yaxis={"visible": False, "fixedrange": False, "scaleanchor": "x"},
        )
        return figure

    def pdf_path(self, zotero_key: str) -> Path | None:
        paper = self.papers_by_key.get(zotero_key)
        if not paper:
            return None
        configured_root = os.environ.get("ZOTERO_STORAGE_DIR")
        storage_root = (
            Path(configured_root)
            if configured_root
            else Path.home() / "Zotero" / "storage"
        )
        return _resolve_pdf_path(
            zotero_key,
            str(paper.get("metadata_attachment_key") or ""),
            str(storage_root),
        )
