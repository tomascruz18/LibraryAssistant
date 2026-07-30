"""Persist normalized paper metadata in a small local SQLite database."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Iterable

from llm import METADATA_PIPELINE_VERSION


DEFAULT_DATABASE_PATH = Path("data/library.sqlite3")
SCHEMA_VERSION = 8


@dataclass(frozen=True)
class ProcessingDecision:
    process: bool
    reason: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def connect_database(
    path: str | Path = DEFAULT_DATABASE_PATH,
) -> sqlite3.Connection:
    """Open the catalog, create its directory, and initialize its schema."""
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    _initialize_schema(connection)
    return connection


def _initialize_schema(connection: sqlite3.Connection) -> None:
    current_version = connection.execute("PRAGMA user_version").fetchone()[0]
    if current_version > SCHEMA_VERSION:
        raise RuntimeError(
            f"Database schema {current_version} is newer than supported version "
            f"{SCHEMA_VERSION}."
        )

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS papers (
            zotero_key TEXT PRIMARY KEY,
            zotero_version INTEGER NOT NULL DEFAULT 0,
            item_type TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL DEFAULT '',
            doi TEXT NOT NULL DEFAULT '',
            url TEXT NOT NULL DEFAULT '',

            zotero_authors_json TEXT NOT NULL DEFAULT '[]',
            zotero_date TEXT NOT NULL DEFAULT '',
            zotero_abstract TEXT NOT NULL DEFAULT '',

            authors_json TEXT NOT NULL DEFAULT '[]',
            publication_date TEXT NOT NULL DEFAULT '',
            abstract TEXT NOT NULL DEFAULT '',
            metadata_source_json TEXT NOT NULL DEFAULT '{}',
            document_type TEXT NOT NULL DEFAULT 'unknown',
            is_supported INTEGER NOT NULL DEFAULT 1
                CHECK (is_supported IN (0, 1)),

            attachment_key TEXT,
            attachment_version INTEGER,
            text_extraction_method TEXT,
            metadata_pipeline_version INTEGER NOT NULL DEFAULT 1,
            llm_model TEXT,
            llm_context_window_tokens INTEGER,
            metadata_llm_attempts INTEGER,
            source_fingerprint TEXT NOT NULL,

            processing_status TEXT NOT NULL
                CHECK (processing_status IN ('complete', 'partial', 'error')),
            metadata_error TEXT,
            first_seen_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            processed_at TEXT NOT NULL,
            is_deleted INTEGER NOT NULL DEFAULT 0
                CHECK (is_deleted IN (0, 1))
        );

        CREATE INDEX IF NOT EXISTS idx_papers_zotero_version
            ON papers(zotero_version);
        CREATE INDEX IF NOT EXISTS idx_papers_status
            ON papers(processing_status);
        CREATE INDEX IF NOT EXISTS idx_papers_title
            ON papers(title);

        CREATE TABLE IF NOT EXISTS sync_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS paper_embeddings (
            zotero_key TEXT PRIMARY KEY
                REFERENCES papers(zotero_key) ON DELETE CASCADE,
            embedding_model TEXT NOT NULL,
            input_hash TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            vector BLOB NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_paper_embeddings_model
            ON paper_embeddings(embedding_model);

        CREATE TABLE IF NOT EXISTS paper_similarity_edges (
            embedding_model TEXT NOT NULL,
            source_key TEXT NOT NULL REFERENCES papers(zotero_key) ON DELETE CASCADE,
            target_key TEXT NOT NULL REFERENCES papers(zotero_key) ON DELETE CASCADE,
            similarity REAL NOT NULL,
            PRIMARY KEY (embedding_model, source_key, target_key),
            CHECK (source_key < target_key)
        );

        CREATE INDEX IF NOT EXISTS idx_similarity_edges_source
            ON paper_similarity_edges(embedding_model, source_key);
        CREATE INDEX IF NOT EXISTS idx_similarity_edges_target
            ON paper_similarity_edges(embedding_model, target_key);

        CREATE TABLE IF NOT EXISTS paper_clusters (
            embedding_model TEXT NOT NULL,
            zotero_key TEXT NOT NULL REFERENCES papers(zotero_key) ON DELETE CASCADE,
            cluster_id INTEGER NOT NULL,
            PRIMARY KEY (embedding_model, zotero_key)
        );

        CREATE INDEX IF NOT EXISTS idx_paper_clusters_cluster
            ON paper_clusters(embedding_model, cluster_id);

        CREATE TABLE IF NOT EXISTS similarity_graph_state (
            embedding_model TEXT PRIMARY KEY,
            input_fingerprint TEXT NOT NULL,
            neighbors INTEGER NOT NULL,
            similarity_percentile REAL NOT NULL,
            similarity_threshold REAL,
            built_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS paper_projections (
            embedding_model TEXT NOT NULL,
            zotero_key TEXT NOT NULL REFERENCES papers(zotero_key) ON DELETE CASCADE,
            x REAL NOT NULL,
            y REAL NOT NULL,
            PRIMARY KEY (embedding_model, zotero_key)
        );

        CREATE TABLE IF NOT EXISTS projection_state (
            embedding_model TEXT PRIMARY KEY,
            input_fingerprint TEXT NOT NULL,
            neighbors INTEGER NOT NULL,
            min_dist REAL NOT NULL,
            random_state INTEGER NOT NULL,
            built_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cluster_labels (
            embedding_model TEXT NOT NULL,
            graph_fingerprint TEXT NOT NULL,
            cluster_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            description TEXT NOT NULL,
            source TEXT NOT NULL CHECK (source IN ('llm', 'title', 'manual')),
            llm_model TEXT,
            label_pipeline_version INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (embedding_model, graph_fingerprint, cluster_id)
        );

        CREATE INDEX IF NOT EXISTS idx_cluster_labels_current
            ON cluster_labels(embedding_model, graph_fingerprint, cluster_id);
        """
    )
    if current_version == 1:
        connection.executescript(
            """
            ALTER TABLE papers
                ADD COLUMN document_type TEXT NOT NULL DEFAULT 'unknown';
            ALTER TABLE papers
                ADD COLUMN is_supported INTEGER NOT NULL DEFAULT 1
                    CHECK (is_supported IN (0, 1));
            """
        )
    if current_version in {1, 2}:
        connection.execute(
            "ALTER TABLE papers ADD COLUMN text_extraction_method TEXT"
        )
    if current_version == 3:
        connection.execute(
            "ALTER TABLE papers ADD COLUMN metadata_llm_attempts INTEGER"
        )
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    connection.commit()


def _source_fingerprint(paper: dict[str, Any]) -> str:
    """Fingerprint inputs that can make stored metadata stale."""
    source = {
        "zotero_key": paper.get("id"),
        "zotero_version": paper.get("zotero_version", 0),
        "zotero_metadata": paper.get("zotero_metadata", {}),
        "attachment_key": paper.get("metadata_attachment_key"),
        "attachment_version": paper.get("metadata_attachment_version"),
        "text_extraction_method": paper.get("text_extraction_method"),
        "metadata_pipeline_version": paper.get(
            "metadata_pipeline_version", METADATA_PIPELINE_VERSION
        ),
        "llm_model": paper.get("llm_model"),
        "llm_context_window_tokens": paper.get("llm_context_window_tokens"),
        "document_type": paper.get("document_type", "unknown"),
    }
    return hashlib.sha256(_json(source).encode("utf-8")).hexdigest()


def _processing_status(paper: dict[str, Any]) -> str:
    if paper.get("metadata_error"):
        return "error"
    if any(not paper.get(field) for field in ("authors", "date", "abstract")):
        return "partial"
    return "complete"


def upsert_paper(
    connection: sqlite3.Connection,
    paper: dict[str, Any],
) -> None:
    """Insert or update one fully processed paper record."""
    now = _utc_now()
    zotero_metadata = paper.get("zotero_metadata", {})
    values = {
        "zotero_key": paper["id"],
        "zotero_version": int(paper.get("zotero_version", 0)),
        "item_type": paper.get("item_type", ""),
        "title": paper.get("title", ""),
        "doi": paper.get("doi", ""),
        "url": paper.get("url", ""),
        "zotero_authors_json": _json(zotero_metadata.get("authors", [])),
        "zotero_date": zotero_metadata.get("date", ""),
        "zotero_abstract": zotero_metadata.get("abstract", ""),
        "authors_json": _json(paper.get("authors", [])),
        "publication_date": paper.get("date", ""),
        "abstract": paper.get("abstract", ""),
        "metadata_source_json": _json(paper.get("metadata_source", {})),
        "document_type": paper.get("document_type", "unknown"),
        "is_supported": int(paper.get("is_supported", True)),
        "attachment_key": paper.get("metadata_attachment_key"),
        "attachment_version": paper.get("metadata_attachment_version"),
        "text_extraction_method": paper.get("text_extraction_method"),
        "metadata_pipeline_version": int(
            paper.get("metadata_pipeline_version", METADATA_PIPELINE_VERSION)
        ),
        "llm_model": paper.get("llm_model"),
        "llm_context_window_tokens": paper.get("llm_context_window_tokens"),
        "metadata_llm_attempts": paper.get("metadata_llm_attempts"),
        "source_fingerprint": _source_fingerprint(paper),
        "processing_status": _processing_status(paper),
        "metadata_error": paper.get("metadata_error"),
        "first_seen_at": now,
        "updated_at": now,
        "processed_at": now,
    }
    connection.execute(
        """
        INSERT INTO papers (
            zotero_key, zotero_version, item_type, title, doi, url,
            zotero_authors_json, zotero_date, zotero_abstract,
            authors_json, publication_date, abstract, metadata_source_json,
            document_type, is_supported,
            attachment_key, attachment_version, text_extraction_method,
            metadata_pipeline_version,
            llm_model, llm_context_window_tokens, metadata_llm_attempts,
            source_fingerprint,
            processing_status, metadata_error, first_seen_at, updated_at,
            processed_at, is_deleted
        ) VALUES (
            :zotero_key, :zotero_version, :item_type, :title, :doi, :url,
            :zotero_authors_json, :zotero_date, :zotero_abstract,
            :authors_json, :publication_date, :abstract, :metadata_source_json,
            :document_type, :is_supported,
            :attachment_key, :attachment_version, :text_extraction_method,
            :metadata_pipeline_version,
            :llm_model, :llm_context_window_tokens, :metadata_llm_attempts,
            :source_fingerprint,
            :processing_status, :metadata_error, :first_seen_at, :updated_at,
            :processed_at, 0
        )
        ON CONFLICT(zotero_key) DO UPDATE SET
            zotero_version = excluded.zotero_version,
            item_type = excluded.item_type,
            title = excluded.title,
            doi = excluded.doi,
            url = excluded.url,
            zotero_authors_json = excluded.zotero_authors_json,
            zotero_date = excluded.zotero_date,
            zotero_abstract = excluded.zotero_abstract,
            authors_json = excluded.authors_json,
            publication_date = excluded.publication_date,
            abstract = excluded.abstract,
            metadata_source_json = excluded.metadata_source_json,
            document_type = excluded.document_type,
            is_supported = excluded.is_supported,
            attachment_key = excluded.attachment_key,
            attachment_version = excluded.attachment_version,
            text_extraction_method = excluded.text_extraction_method,
            metadata_pipeline_version = excluded.metadata_pipeline_version,
            llm_model = excluded.llm_model,
            llm_context_window_tokens = excluded.llm_context_window_tokens,
            metadata_llm_attempts = excluded.metadata_llm_attempts,
            source_fingerprint = excluded.source_fingerprint,
            processing_status = excluded.processing_status,
            metadata_error = excluded.metadata_error,
            updated_at = excluded.updated_at,
            processed_at = excluded.processed_at,
            is_deleted = 0
        """,
        values,
    )


def save_papers(
    papers: Iterable[dict[str, Any]],
    path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    show_progress: bool = False,
) -> int:
    """Save records in one transaction, optionally showing terminal progress."""
    total = _iterable_length(papers) if show_progress else None
    count = 0
    if show_progress:
        _render_save_progress(count, total)
    with closing(connect_database(path)) as connection:
        with connection:
            for paper in papers:
                upsert_paper(connection, paper)
                count += 1
                if show_progress:
                    _render_save_progress(count, total)
    if show_progress:
        print(file=sys.stdout)
    return count


def _iterable_length(items: Iterable[Any]) -> int | None:
    """Return a known iterable length without consuming generators."""
    try:
        return len(items)  # type: ignore[arg-type]
    except TypeError:
        return None


def _render_save_progress(completed: int, total: int | None) -> None:
    """Render one lightweight, dependency-free terminal progress bar."""
    if total is None:
        message = f"Saving papers: {completed}"
    elif total == 0:
        message = "Saving papers: 0/0"
    else:
        width = 24
        filled = round(width * completed / total)
        bar = "#" * filled + "-" * (width - filled)
        message = f"Saving papers: [{bar}] {completed}/{total}"
    print(f"\r{message}", end="", file=sys.stdout, flush=True)


def _decode_paper(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["zotero_key"],
        "zotero_version": row["zotero_version"],
        "item_type": row["item_type"],
        "title": row["title"],
        "doi": row["doi"],
        "url": row["url"],
        "authors": json.loads(row["authors_json"]),
        "date": row["publication_date"],
        "abstract": row["abstract"],
        "metadata_source": json.loads(row["metadata_source_json"]),
        "document_type": row["document_type"],
        "is_supported": bool(row["is_supported"]),
        "zotero_metadata": {
            "authors": json.loads(row["zotero_authors_json"]),
            "date": row["zotero_date"],
            "abstract": row["zotero_abstract"],
        },
        "metadata_attachment_key": row["attachment_key"],
        "metadata_attachment_version": row["attachment_version"],
        "text_extraction_method": row["text_extraction_method"],
        "metadata_pipeline_version": row["metadata_pipeline_version"],
        "llm_model": row["llm_model"],
        "llm_context_window_tokens": row["llm_context_window_tokens"],
        "metadata_llm_attempts": row["metadata_llm_attempts"],
        "processing_status": row["processing_status"],
        "metadata_error": row["metadata_error"],
        "source_fingerprint": row["source_fingerprint"],
        "first_seen_at": row["first_seen_at"],
        "updated_at": row["updated_at"],
        "processed_at": row["processed_at"],
        "is_deleted": bool(row["is_deleted"]),
    }


def get_paper(
    zotero_key: str,
    path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any] | None:
    with closing(connect_database(path)) as connection:
        row = connection.execute(
            "SELECT * FROM papers WHERE zotero_key = ?", (zotero_key,)
        ).fetchone()
    return _decode_paper(row) if row else None


def load_papers(
    path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    include_deleted: bool = False,
) -> list[dict[str, Any]]:
    where = "" if include_deleted else "WHERE is_deleted = 0"
    with closing(connect_database(path)) as connection:
        rows = connection.execute(
            f"SELECT * FROM papers {where} ORDER BY title"
        ).fetchall()
    return [_decode_paper(row) for row in rows]


def needs_processing(
    paper: dict[str, Any],
    path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    pipeline_version: int = METADATA_PIPELINE_VERSION,
    retry_errors: bool = False,
) -> ProcessingDecision:
    """Compare a current Zotero record with its stored processing signature."""
    with closing(connect_database(path)) as connection:
        row = connection.execute(
            """
            SELECT zotero_version, attachment_key, attachment_version,
                   metadata_pipeline_version, llm_model,
                   llm_context_window_tokens, metadata_source_json,
                   processing_status, is_deleted
            FROM papers WHERE zotero_key = ?
            """,
            (paper["id"],),
        ).fetchone()

    if row is None:
        return ProcessingDecision(True, "new_item")
    if row["is_deleted"]:
        return ProcessingDecision(True, "restored_item")
    if int(paper.get("zotero_version", 0)) != row["zotero_version"]:
        return ProcessingDecision(True, "zotero_item_modified")

    incoming_attachment_version = paper.get("metadata_attachment_version")
    if (
        incoming_attachment_version is not None
        and incoming_attachment_version != row["attachment_version"]
    ):
        return ProcessingDecision(True, "pdf_attachment_modified")

    sources = json.loads(row["metadata_source_json"])
    uses_llm = any(str(source).startswith("llm_") for source in sources.values())
    if uses_llm and row["metadata_pipeline_version"] != pipeline_version:
        return ProcessingDecision(True, "llm_pipeline_changed")
    if uses_llm and (
        row["llm_model"] != paper.get("llm_model")
        or row["llm_context_window_tokens"]
        != paper.get("llm_context_window_tokens")
    ):
        return ProcessingDecision(True, "llm_configuration_changed")
    if row["processing_status"] == "error" and retry_errors:
        return ProcessingDecision(True, "retry_previous_error")

    return ProcessingDecision(False, "unchanged")


def set_sync_state(
    key: str,
    value: str,
    path: str | Path = DEFAULT_DATABASE_PATH,
) -> None:
    now = _utc_now()
    with closing(connect_database(path)) as connection:
        with connection:
            connection.execute(
                """
                INSERT INTO sync_state(key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, now),
            )


def get_sync_state(
    key: str,
    path: str | Path = DEFAULT_DATABASE_PATH,
) -> str | None:
    with closing(connect_database(path)) as connection:
        row = connection.execute(
            "SELECT value FROM sync_state WHERE key = ?", (key,)
        ).fetchone()
    return row["value"] if row else None
