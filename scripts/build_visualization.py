"""Build or reuse UMAP coordinates and write the interactive paper-map HTML file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embeddings import DEFAULT_MODEL
from storage import DEFAULT_DATABASE_PATH
from visualization import (
    DEFAULT_UMAP_MIN_DIST,
    DEFAULT_UMAP_NEIGHBORS,
    DEFAULT_UMAP_RANDOM_STATE,
    sync_database_projection,
    write_database_force_visualization,
    write_database_visualization,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cached UMAP coordinates and an interactive similarity-map HTML file."
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--neighbors", type=int, default=DEFAULT_UMAP_NEIGHBORS)
    parser.add_argument("--min-dist", type=float, default=DEFAULT_UMAP_MIN_DIST)
    parser.add_argument("--random-state", type=int, default=DEFAULT_UMAP_RANDOM_STATE)
    parser.add_argument(
        "--output", type=Path, default=Path("data/library_map.html"),
        help="UMAP HTML output path",
    )
    parser.add_argument(
        "--force-output", type=Path, default=Path("data/library_force_graph.html"),
        help="Force-layout HTML output path",
    )
    arguments = parser.parse_args()

    result = sync_database_projection(
        arguments.database,
        model_name=arguments.model,
        neighbors=arguments.neighbors,
        min_dist=arguments.min_dist,
        random_state=arguments.random_state,
    )
    output = write_database_visualization(
        arguments.output, arguments.database, model_name=arguments.model
    )
    force_output = write_database_force_visualization(
        arguments.force_output, arguments.database, model_name=arguments.model
    )
    print("UMAP rebuilt:" if result.rebuilt else "UMAP reused:", result.rebuilt)
    print(f"Papers projected: {result.papers}")
    print(f"Visualization: {output.resolve()}")
    print(f"Force graph: {force_output.resolve()}")


if __name__ == "__main__":
    main()
