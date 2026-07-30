"""Run the complete LibraryAssistant pipeline with one command."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sqlite3
import sys
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embeddings import DEFAULT_MODEL as DEFAULT_EMBEDDING_MODEL
from clusters import DEFAULT_REPRESENTATIVE_PAPERS
from graph import DEFAULT_NEIGHBORS, DEFAULT_SIMILARITY_PERCENTILE
from llm import (
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    DEFAULT_MODEL as DEFAULT_LLM_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_SUMMARIZATION_TOKENS,
)
from pipeline import run_pipeline
from storage import DEFAULT_DATABASE_PATH
from visualization import (
    DEFAULT_UMAP_MIN_DIST,
    DEFAULT_UMAP_NEIGHBORS,
    DEFAULT_UMAP_RANDOM_STATE,
)


def _database_sidecars(path: Path) -> tuple[Path, Path]:
    return Path(f"{path}-wal"), Path(f"{path}-shm")


def _checkpoint_database(path: Path) -> None:
    if not path.exists():
        return
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()


def _remove_database_family(path: Path) -> None:
    path.unlink(missing_ok=True)
    for sidecar in _database_sidecars(path):
        sidecar.unlink(missing_ok=True)


def _promote_rebuilt_database(temporary: Path, target: Path) -> None:
    """Atomically promote a successful rebuild while retaining rollback safety."""
    _checkpoint_database(temporary)
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.with_name(f".{target.name}.{uuid4().hex}.backup")
    had_existing = target.exists()
    if had_existing:
        _checkpoint_database(target)
        os.replace(target, backup)
        for sidecar in _database_sidecars(target):
            sidecar.unlink(missing_ok=True)
    try:
        os.replace(temporary, target)
    except Exception:
        if had_existing and backup.exists():
            os.replace(backup, target)
        raise
    else:
        backup.unlink(missing_ok=True)
    finally:
        for sidecar in _database_sidecars(temporary):
            sidecar.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Zotero metadata, embeddings, graph/clustering, UMAP, and HTML outputs."
    )
    parser.add_argument("--papers", type=int, default=1_000, help="Maximum Zotero papers")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--rebuild", action="store_true",
        help="Build a new database from scratch and replace the old one only after success",
    )
    source.add_argument(
        "--database-only", action="store_true",
        help="Skip Zotero/LLM work and finish downstream stages from the existing database",
    )
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--max-paper-tokens", type=int, default=MAX_SUMMARIZATION_TOKENS)
    parser.add_argument("--front-matter-characters", type=int, default=12_000)
    parser.add_argument("--llm-timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--llm-context", type=int, default=DEFAULT_CONTEXT_WINDOW_TOKENS)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument(
        "--similarity-percentile", type=float, default=DEFAULT_SIMILARITY_PERCENTILE
    )
    parser.add_argument("--neighbors", type=int, default=DEFAULT_NEIGHBORS)
    parser.add_argument(
        "--skip-cluster-labels", action="store_true",
        help="Do not call the LLM to name Leiden clusters",
    )
    parser.add_argument(
        "--cluster-label-representatives",
        type=int,
        default=DEFAULT_REPRESENTATIVE_PAPERS,
        help="Central papers supplied to the LLM for each multi-paper cluster",
    )
    parser.add_argument("--umap-neighbors", type=int, default=DEFAULT_UMAP_NEIGHBORS)
    parser.add_argument("--umap-min-dist", type=float, default=DEFAULT_UMAP_MIN_DIST)
    parser.add_argument("--umap-random-state", type=int, default=DEFAULT_UMAP_RANDOM_STATE)
    parser.add_argument("--umap-output", type=Path, default=Path("data/library_map.html"))
    parser.add_argument(
        "--force-output", type=Path, default=Path("data/library_force_graph.html")
    )
    parser.add_argument("--no-progress", action="store_true")
    arguments = parser.parse_args()

    if arguments.papers < 1:
        parser.error("--papers must be at least 1")
    if arguments.max_paper_tokens < 1:
        parser.error("--max-paper-tokens must be at least 1")
    if not 0 < arguments.similarity_percentile < 100:
        parser.error("--similarity-percentile must be between 0 and 100")
    if arguments.neighbors < 1:
        parser.error("--neighbors must be at least 1")
    if arguments.cluster_label_representatives < 1:
        parser.error("--cluster-label-representatives must be at least 1")

    target_database = arguments.database.resolve()
    run_database = target_database
    if arguments.rebuild:
        target_database.parent.mkdir(parents=True, exist_ok=True)
        run_database = target_database.with_name(
            f".{target_database.name}.{uuid4().hex}.rebuild"
        )

    try:
        result = run_pipeline(
            database_path=run_database,
            paper_limit=arguments.papers,
            refresh_zotero=not arguments.database_only,
            llm_model=arguments.llm_model,
            embedding_model=arguments.embedding_model,
            max_metadata_characters=arguments.front_matter_characters,
            max_paper_tokens=arguments.max_paper_tokens,
            llm_timeout_seconds=arguments.llm_timeout,
            llm_context_window_tokens=arguments.llm_context,
            embedding_batch_size=arguments.embedding_batch_size,
            similarity_percentile=arguments.similarity_percentile,
            graph_neighbors=arguments.neighbors,
            label_clusters=not arguments.skip_cluster_labels,
            cluster_label_representatives=arguments.cluster_label_representatives,
            umap_neighbors=arguments.umap_neighbors,
            umap_min_dist=arguments.umap_min_dist,
            umap_random_state=arguments.umap_random_state,
            umap_output=arguments.umap_output,
            force_output=arguments.force_output,
            show_progress=not arguments.no_progress,
        )
        if arguments.rebuild:
            _promote_rebuilt_database(run_database, target_database)
    except Exception:
        if arguments.rebuild:
            _remove_database_family(run_database)
        raise

    print("\nPipeline complete")
    print(f"Papers saved: {result.papers_saved}")
    print(
        f"Embeddings: {result.embeddings.generated} generated, "
        f"{result.embeddings.reused} reused"
    )
    print(
        f"Graph: {result.graph.edges} edges, {result.graph.clusters} clusters, "
        f"rebuilt={result.graph.rebuilt}"
    )
    if result.labels is not None:
        print(
            f"Cluster labels: {result.labels.generated} generated, "
            f"{result.labels.reused} reused, {result.labels.title_based} title-based"
        )
    print(f"UMAP rebuilt: {result.projection.rebuilt}")
    print(f"Database: {target_database}")
    print(f"UMAP HTML: {result.umap_output.resolve()}")
    print(f"Force HTML: {result.force_output.resolve()}")


if __name__ == "__main__":
    main()
