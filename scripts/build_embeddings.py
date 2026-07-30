"""Build or incrementally update embeddings stored in the local SQLite catalog.

Example:
    python scripts/build_embeddings.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embeddings import DEFAULT_MODEL, sync_database_embeddings
from storage import DEFAULT_DATABASE_PATH


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build only missing or stale embeddings from the SQLite paper catalog."
    )
    parser.add_argument(
        "--database", type=Path, default=DEFAULT_DATABASE_PATH, help="SQLite catalog path"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence Transformer model")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")
    arguments = parser.parse_args()

    result = sync_database_embeddings(
        arguments.database, model_name=arguments.model, batch_size=arguments.batch_size
    )
    print(f"Papers considered: {result.total_papers}")
    print(f"Embeddings generated: {result.generated}")
    print(f"Embeddings reused: {result.reused}")
    print(f"Papers skipped: {result.skipped}")


if __name__ == "__main__":
    main()
