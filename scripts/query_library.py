"""Query persisted paper embeddings, graph neighbours, and Leiden clusters.

Examples:
    python scripts/query_library.py
    python scripts/query_library.py --query "regenerative cooling"
    python scripts/query_library.py --neighbors 37DLSCKU
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import sys
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embeddings import DEFAULT_MODEL, load_database_embedding_index, load_embedding_model, paper_text
from clusters import current_graph_fingerprint
from search import bm25_search, build_bm25, hybrid_search, semantic_search
from storage import DEFAULT_DATABASE_PATH


def _print_results(results: Sequence[dict[str, Any]]) -> None:
    if not results:
        print("No results.")
        return
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank:2d}. [{result['score']:.3f}] "
            f"(semantic={result.get('semantic_score', 0):.3f}, "
            f"bm25={result.get('bm25_score', 0):.3f}) "
            f"{result['id']} — {result.get('title', 'Untitled')}"
        )


def _nearest_papers(
    zotero_key: str,
    papers: Sequence[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    indices = {paper["id"]: index for index, paper in enumerate(papers)}
    index = indices.get(zotero_key.upper())
    if index is None:
        raise KeyError(f"No embedded paper found with Zotero key {zotero_key!r}.")
    scores = embeddings[index] @ embeddings.T
    order = [candidate for candidate in np.argsort(scores)[::-1] if candidate != index]
    return [
        {
            **papers[int(candidate)],
            "score": float(scores[candidate]),
            "semantic_score": float(scores[candidate]),
            "bm25_score": 0.0,
        }
        for candidate in order[:limit]
    ]


def _stored_graph_neighbours(
    database: Path,
    model_name: str,
    zotero_key: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT CASE WHEN edge.source_key = ? THEN edge.target_key ELSE edge.source_key END
                       AS zotero_key,
                   edge.similarity, paper.title
            FROM paper_similarity_edges AS edge
            JOIN papers AS paper ON paper.zotero_key =
                CASE WHEN edge.source_key = ? THEN edge.target_key ELSE edge.source_key END
            WHERE edge.embedding_model = ?
              AND (edge.source_key = ? OR edge.target_key = ?)
            ORDER BY edge.similarity DESC
            LIMIT ?
            """,
            (zotero_key.upper(), zotero_key.upper(), model_name, zotero_key.upper(), zotero_key.upper(), limit),
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "id": row["zotero_key"],
            "title": row["title"],
            "score": float(row["similarity"]),
            "semantic_score": float(row["similarity"]),
            "bm25_score": 0.0,
        }
        for row in rows
    ]


def _print_clusters(database: Path, model_name: str) -> None:
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            """
            SELECT cluster.cluster_id, COUNT(*)
            FROM paper_clusters AS cluster
            WHERE cluster.embedding_model = ?
            GROUP BY cluster.cluster_id
            ORDER BY COUNT(*) DESC, cluster.cluster_id
            """,
            (model_name,),
        ).fetchall()
        if rows:
            fingerprint = current_graph_fingerprint(database, model_name)
            label_rows = connection.execute(
                """
                SELECT cluster_id, label FROM cluster_labels
                WHERE embedding_model = ? AND graph_fingerprint = ?
                """,
                (model_name, fingerprint),
            ).fetchall()
        else:
            label_rows = []
    finally:
        connection.close()
    if not rows:
        print("No stored clusters. Run scripts\\build_graph.py first.")
        return
    labels = {cluster_id: label for cluster_id, label in label_rows}
    for cluster_id, count in rows:
        label = labels.get(cluster_id)
        name = f" — {label}" if label else ""
        print(f"Cluster {cluster_id}{name}: {count} papers")


def _print_cluster_papers(
    database: Path, model_name: str, cluster_id: int) -> None:
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            """
            SELECT paper.zotero_key, paper.title
            FROM paper_clusters AS cluster
            JOIN papers AS paper ON paper.zotero_key = cluster.zotero_key
            WHERE cluster.embedding_model = ? AND cluster.cluster_id = ?
            ORDER BY paper.title COLLATE NOCASE
            """,
            (model_name, cluster_id),
        ).fetchall()
    finally:
        connection.close()
    for key, title in rows:
        print(f"{key} — {title}")
    if not rows:
        print(f"Cluster {cluster_id} was not found.")


def _run_search(
    query: str,
    mode: str,
    model: Any,
    embeddings: np.ndarray,
    papers: Sequence[dict[str, Any]],
    bm25: Any,
    limit: int,
) -> None:
    if mode == "semantic":
        results = semantic_search(query, model, embeddings, papers, k=limit)
    elif mode == "bm25":
        results = bm25_search(query, bm25, papers, k=limit)
    else:
        results = hybrid_search(query, model, embeddings, bm25, papers, k=limit)
    _print_results(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the persisted LibraryAssistant catalog.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--query", help="Run one semantic, BM25, or hybrid query and exit")
    parser.add_argument(
        "--mode", choices=("hybrid", "semantic", "bm25"), default="hybrid"
    )
    parser.add_argument("--neighbors", metavar="ZOTERO_KEY", help="Show nearest papers and exit")
    parser.add_argument(
        "--graph-neighbors", metavar="ZOTERO_KEY", help="Show saved graph neighbours and exit"
    )
    parser.add_argument("--cluster", type=int, help="List one Leiden cluster and exit")
    parser.add_argument("--clusters", action="store_true", help="List Leiden cluster sizes and exit")
    arguments = parser.parse_args()
    if arguments.limit < 1:
        parser.error("--limit must be at least 1")
    if not arguments.database.is_file():
        parser.error(f"Database not found: {arguments.database}")

    if arguments.clusters:
        _print_clusters(arguments.database, arguments.model)
        return
    if arguments.cluster is not None:
        _print_cluster_papers(arguments.database, arguments.model, arguments.cluster)
        return
    if arguments.graph_neighbors:
        _print_results(
            _stored_graph_neighbours(
                arguments.database, arguments.model, arguments.graph_neighbors,
                limit=arguments.limit,
            )
        )
        return

    papers, embeddings = load_database_embedding_index(
        arguments.database, model_name=arguments.model
    )
    if not papers:
        parser.error("No stored embeddings. Run scripts\\build_embeddings.py first.")
    if arguments.neighbors:
        _print_results(
            _nearest_papers(arguments.neighbors, papers, embeddings, limit=arguments.limit)
        )
        return

    bm25 = build_bm25([paper_text(paper) for paper in papers])
    if arguments.query:
        model = None if arguments.mode == "bm25" else load_embedding_model(arguments.model)
        _run_search(
            arguments.query, arguments.mode, model, embeddings, papers, bm25, arguments.limit
        )
        return

    model = load_embedding_model(arguments.model)
    print("Ready. Commands: q <query>, s <query>, b <query>, n <ZOTERO_KEY>,")
    print("                g <ZOTERO_KEY>, clusters, c <cluster_id>, x")
    while True:
        command = input("\nQuery: ").strip()
        if command.lower() == "x":
            return
        if command.startswith("q "):
            _run_search(command[2:].strip(), "hybrid", model, embeddings, papers, bm25, arguments.limit)
        elif command.startswith("s "):
            _run_search(command[2:].strip(), "semantic", model, embeddings, papers, bm25, arguments.limit)
        elif command.startswith("b "):
            _run_search(command[2:].strip(), "bm25", model, embeddings, papers, bm25, arguments.limit)
        elif command.startswith("n "):
            try:
                _print_results(_nearest_papers(command[2:].strip(), papers, embeddings, limit=arguments.limit))
            except KeyError as error:
                print(error)
        elif command.startswith("g "):
            _print_results(_stored_graph_neighbours(arguments.database, arguments.model, command[2:].strip(), limit=arguments.limit))
        elif command == "clusters":
            _print_clusters(arguments.database, arguments.model)
        elif command.startswith("c "):
            try:
                _print_cluster_papers(arguments.database, arguments.model, int(command[2:].strip()))
            except ValueError:
                print("Cluster ID must be an integer.")
        else:
            print("Unknown command.")


if __name__ == "__main__":
    main()
