"""Run the complete local LibraryAssistant data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from data import load_zotero_library
from clusters import (
    DEFAULT_REPRESENTATIVE_PAPERS,
    ClusterLabelSyncResult,
    label_database_clusters,
)
from embeddings import DEFAULT_MODEL as DEFAULT_EMBEDDING_MODEL
from embeddings import EmbeddingSyncResult, sync_database_embeddings
from graph import (
    DEFAULT_NEIGHBORS,
    DEFAULT_SIMILARITY_PERCENTILE,
    GraphSyncResult,
    sync_database_graph,
)
from llm import (
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    DEFAULT_MODEL as DEFAULT_LLM_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_SUMMARIZATION_TOKENS,
)
from storage import DEFAULT_DATABASE_PATH, load_papers, save_papers
from visualization import (
    DEFAULT_UMAP_MIN_DIST,
    DEFAULT_UMAP_NEIGHBORS,
    DEFAULT_UMAP_RANDOM_STATE,
    ProjectionSyncResult,
    sync_database_projection,
    write_database_force_visualization,
    write_database_visualization,
)


@dataclass(frozen=True)
class PipelineResult:
    papers_saved: int
    embeddings: EmbeddingSyncResult
    graph: GraphSyncResult
    labels: ClusterLabelSyncResult | None
    projection: ProjectionSyncResult
    umap_output: Path
    force_output: Path


def run_pipeline(
    *,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    paper_limit: int = 1_000,
    refresh_zotero: bool = True,
    llm_model: str = DEFAULT_LLM_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    max_metadata_characters: int = 12_000,
    max_paper_tokens: int = MAX_SUMMARIZATION_TOKENS,
    llm_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    llm_context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
    embedding_batch_size: int = 16,
    similarity_percentile: float = DEFAULT_SIMILARITY_PERCENTILE,
    graph_neighbors: int = DEFAULT_NEIGHBORS,
    label_clusters: bool = True,
    cluster_label_representatives: int = DEFAULT_REPRESENTATIVE_PAPERS,
    umap_neighbors: int = DEFAULT_UMAP_NEIGHBORS,
    umap_min_dist: float = DEFAULT_UMAP_MIN_DIST,
    umap_random_state: int = DEFAULT_UMAP_RANDOM_STATE,
    umap_output: str | Path = "data/library_map.html",
    force_output: str | Path = "data/library_force_graph.html",
    show_progress: bool = True,
    report: Callable[[str], None] = print,
) -> PipelineResult:
    """Run metadata, embeddings, graph/clustering, UMAP, and visualization."""
    if paper_limit < 1:
        raise ValueError("paper_limit must be at least 1.")

    database = Path(database_path)
    if refresh_zotero:
        report("[1/6] Loading and enriching Zotero papers")
        papers = load_zotero_library(
            limit=paper_limit,
            llm_model=llm_model,
            max_metadata_characters=max_metadata_characters,
            llm_timeout_seconds=llm_timeout_seconds,
            llm_context_window_tokens=llm_context_window_tokens,
            max_summary_tokens=max_paper_tokens,
            show_progress=show_progress,
        )
        papers_saved = save_papers(
            papers, database, show_progress=show_progress
        )
    else:
        report("[1/6] Reusing papers already stored in SQLite")
        papers_saved = len(load_papers(database))
        if papers_saved == 0:
            raise ValueError("The database contains no papers to process.")

    report("[2/6] Building or reusing embeddings")
    embedding_result = sync_database_embeddings(
        database, model_name=embedding_model, batch_size=embedding_batch_size
    )

    report("[3/6] Building or reusing similarity graph and Leiden clusters")
    graph_result = sync_database_graph(
        database,
        model_name=embedding_model,
        neighbors=graph_neighbors,
        percentile=similarity_percentile,
    )

    label_result = None
    if label_clusters:
        report("[4/6] Generating or reusing cluster labels")
        label_result = label_database_clusters(
            database,
            embedding_model=embedding_model,
            llm_model=llm_model,
            representative_papers=cluster_label_representatives,
            timeout_seconds=llm_timeout_seconds,
            context_window_tokens=llm_context_window_tokens,
        )
    else:
        report("[4/6] Skipping cluster labels")

    report("[5/6] Building or reusing UMAP coordinates")
    projection_result = sync_database_projection(
        database,
        model_name=embedding_model,
        neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        random_state=umap_random_state,
    )

    report("[6/6] Writing UMAP and force-layout HTML visualizations")
    umap_path = write_database_visualization(
        umap_output, database, model_name=embedding_model
    )
    force_path = write_database_force_visualization(
        force_output, database, model_name=embedding_model
    )
    return PipelineResult(
        papers_saved=papers_saved,
        embeddings=embedding_result,
        graph=graph_result,
        labels=label_result,
        projection=projection_result,
        umap_output=umap_path,
        force_output=force_path,
    )
