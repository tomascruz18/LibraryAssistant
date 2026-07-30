"""Build or reuse the persisted paper-similarity graph and Leiden clusters."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embeddings import DEFAULT_MODEL
from graph import (
    DEFAULT_NEIGHBORS,
    DEFAULT_SIMILARITY_PERCENTILE,
    sync_database_graph,
)
from storage import DEFAULT_DATABASE_PATH


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a 90th-percentile, top-k similarity graph and Leiden clusters."
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--neighbors", type=int, default=DEFAULT_NEIGHBORS)
    parser.add_argument(
        "--percentile", type=float, default=DEFAULT_SIMILARITY_PERCENTILE
    )
    arguments = parser.parse_args()

    result = sync_database_graph(
        arguments.database,
        model_name=arguments.model,
        neighbors=arguments.neighbors,
        percentile=arguments.percentile,
    )
    print("Graph rebuilt:" if result.rebuilt else "Graph reused:", result.rebuilt)
    print(f"Papers: {result.papers}")
    print(f"Edges: {result.edges}")
    print(f"Clusters: {result.clusters}")
    if result.similarity_threshold is not None:
        print(f"Similarity threshold: {result.similarity_threshold:.4f}")


if __name__ == "__main__":
    main()
