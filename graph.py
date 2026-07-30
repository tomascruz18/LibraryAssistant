"""Build and cluster a paper-similarity graph."""

from __future__ import annotations

from typing import Any, Sequence

import igraph as ig
import leidenalg
import numpy as np


def build_similarity_graph(
    embeddings: np.ndarray,
    papers: Sequence[dict[str, Any]] | None = None,
    *,
    neighbors: int = 10,
    threshold: float = 0.55,
) -> ig.Graph:
    """Build a deduplicated undirected k-nearest-neighbour graph."""
    similarity = embeddings @ embeddings.T
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

    if papers is not None:
        graph.vs["document_id"] = [
            str(paper.get("id", index)) for index, paper in enumerate(papers)
        ]
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
