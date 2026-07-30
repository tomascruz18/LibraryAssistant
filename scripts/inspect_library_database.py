"""Inspect the local LibraryAssistant SQLite catalog without modifying it.

Examples:
    python scripts/inspect_library_database.py
    python scripts/inspect_library_database.py --errors
    python scripts/inspect_library_database.py --llm
    python scripts/inspect_library_database.py --status complete --limit 50
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
from typing import Any


DEFAULT_DATABASE = Path("data/library.sqlite3")


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    if not rows:
        print("  No matching records.")
        return

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render(row: list[str]) -> str:
        return "  " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        )

    print(render(headers))
    print("  " + "-+-".join("-" * width for width in widths))
    for row in rows:
        print(render(row))


def _metadata_source(row: sqlite3.Row, field: str) -> str:
    sources = json.loads(row["metadata_source_json"])
    return str(sources.get(field) or "missing")


def _format_size(size: int) -> str:
    units = ("B", "KB", "MB", "GB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def _fetch_rows(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT zotero_key, title, processing_status, metadata_error,
               metadata_source_json, document_type, is_supported,
               text_extraction_method, updated_at
        FROM papers
        WHERE is_deleted = 0
        ORDER BY title COLLATE NOCASE
        """
    ).fetchall()


def print_summary(database_path: Path, connection: sqlite3.Connection) -> None:
    rows = _fetch_rows(connection)
    status_counts = Counter(row["processing_status"] for row in rows)
    abstract_sources = Counter(_metadata_source(row, "abstract") for row in rows)
    extraction_methods = Counter(
        row["text_extraction_method"] or "not needed" for row in rows
    )
    document_types = Counter(row["document_type"] for row in rows)

    print(f"Database: {database_path.resolve()}")
    print(f"Size: {_format_size(database_path.stat().st_size)}")
    print(f"Integrity: {connection.execute('PRAGMA integrity_check').fetchone()[0]}")
    print(f"Schema version: {connection.execute('PRAGMA user_version').fetchone()[0]}")
    print(f"Active records: {len(rows)}")

    print("\nProcessing status")
    _print_table(
        ["Status", "Count"],
        [[status, str(count)] for status, count in sorted(status_counts.items())],
    )

    print("\nAbstract source")
    _print_table(
        ["Source", "Count"],
        [[source, str(count)] for source, count in sorted(abstract_sources.items())],
    )

    print("\nPDF extraction method")
    _print_table(
        ["Method", "Count"],
        [[method, str(count)] for method, count in sorted(extraction_methods.items())],
    )

    print("\nDocument type")
    _print_table(
        ["Type", "Count"],
        [[kind, str(count)] for kind, count in sorted(document_types.items())],
    )


def print_records(
    connection: sqlite3.Connection,
    *,
    heading: str,
    where_clause: str,
    parameters: tuple[Any, ...] = (),
    limit: int,
) -> None:
    rows = connection.execute(
        f"""
        SELECT zotero_key, title, processing_status, metadata_error,
               metadata_source_json, document_type, text_extraction_method
        FROM papers
        WHERE is_deleted = 0 AND ({where_clause})
        ORDER BY title COLLATE NOCASE
        LIMIT ?
        """,
        (*parameters, limit),
    ).fetchall()
    print(f"\n{heading} (up to {limit})")
    _print_table(
        ["Key", "Title", "Status", "Abstract source", "Type", "PDF text", "Error"],
        [
            [
                row["zotero_key"],
                row["title"],
                row["processing_status"],
                _metadata_source(row, "abstract"),
                row["document_type"],
                row["text_extraction_method"] or "-",
                row["metadata_error"] or "-",
            ]
            for row in rows
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect LibraryAssistant's local SQLite metadata catalog."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help="Path to the SQLite catalog (default: data/library.sqlite3)",
    )
    parser.add_argument("--errors", action="store_true", help="List error records")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="List records whose abstract came from the LLM",
    )
    parser.add_argument(
        "--status",
        choices=("complete", "partial", "error"),
        help="List records with this processing status",
    )
    parser.add_argument(
        "--source",
        choices=("zotero", "llm_extracted", "llm_summary", "missing"),
        help="List records with this abstract source",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum records to print for each listing (default: 20)",
    )
    arguments = parser.parse_args()

    if arguments.limit < 1:
        parser.error("--limit must be at least 1")
    if not arguments.database.is_file():
        parser.error(f"Database not found: {arguments.database}")

    connection = sqlite3.connect(arguments.database)
    connection.row_factory = sqlite3.Row
    try:
        print_summary(arguments.database, connection)
        if arguments.errors:
            print_records(
                connection,
                heading="Error records",
                where_clause="processing_status = 'error'",
                limit=arguments.limit,
            )
        if arguments.llm:
            print_records(
                connection,
                heading="LLM-enriched records",
                where_clause="json_extract(metadata_source_json, '$.abstract') IN (?, ?)",
                parameters=("llm_extracted", "llm_summary"),
                limit=arguments.limit,
            )
        if arguments.status:
            print_records(
                connection,
                heading=f"Records with status '{arguments.status}'",
                where_clause="processing_status = ?",
                parameters=(arguments.status,),
                limit=arguments.limit,
            )
        if arguments.source:
            print_records(
                connection,
                heading=f"Records with abstract source '{arguments.source}'",
                where_clause="COALESCE(json_extract(metadata_source_json, '$.abstract'), 'missing') = ?",
                parameters=(arguments.source,),
                limit=arguments.limit,
            )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
