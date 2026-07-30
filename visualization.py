"""UMAP projection and Plotly visualizations."""

from __future__ import annotations

from typing import Any, Sequence

import igraph as ig
import numpy as np
import plotly.graph_objects as go
import umap


def project_embeddings(
    embeddings: np.ndarray,
    *,
    random_state: int = 42,
) -> np.ndarray:
    return umap.UMAP(random_state=random_state).fit_transform(embeddings)


def plot_papers(
    coordinates: np.ndarray,
    papers: Sequence[dict[str, Any]],
    communities: Sequence[int] | None = None,
    graph: ig.Graph | None = None,
) -> go.Figure:
    """Create an interactive paper map, optionally including graph edges."""
    traces: list[go.Scatter] = []

    if graph is not None:
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        for edge in graph.es:
            source, target = edge.tuple
            edge_x.extend([coordinates[source, 0], coordinates[target, 0], None])
            edge_y.extend([coordinates[source, 1], coordinates[target, 1], None])
        traces.append(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"width": 0.5, "color": "lightgray"},
                hoverinfo="none",
            )
        )

    colors = list(communities) if communities is not None else [0] * len(papers)
    traces.append(
        go.Scatter(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            mode="markers",
            text=[paper.get("title", "") for paper in papers],
            hovertemplate="<b>%{text}</b><extra></extra>",
            marker={
                "size": 10,
                "color": colors,
                "colorscale": "Viridis",
                "line": {"width": 0.5, "color": "black"},
            },
        )
    )

    figure = go.Figure(data=traces)
    figure.update_layout(
        showlegend=False,
        hovermode="closest",
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure
