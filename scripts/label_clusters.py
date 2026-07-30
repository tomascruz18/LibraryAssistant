"""Generate or reuse concise names for the current saved Leiden clusters."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clusters import DEFAULT_REPRESENTATIVE_PAPERS, label_database_clusters
from embeddings import DEFAULT_MODEL as DEFAULT_EMBEDDING_MODEL
from llm import DEFAULT_CONTEXT_WINDOW_TOKENS, DEFAULT_MODEL, DEFAULT_TIMEOUT_SECONDS
from storage import DEFAULT_DATABASE_PATH


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Name current Leiden clusters using central papers and a local LLM."
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--llm-model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--representatives", type=int, default=DEFAULT_REPRESENTATIVE_PAPERS
    )
    parser.add_argument("--llm-timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--llm-context", type=int, default=DEFAULT_CONTEXT_WINDOW_TOKENS)
    parser.add_argument("--refresh", action="store_true", help="Regenerate existing automatic labels")
    arguments = parser.parse_args()
    if arguments.representatives < 1:
        parser.error("--representatives must be at least 1")

    result = label_database_clusters(
        arguments.database,
        embedding_model=arguments.embedding_model,
        llm_model=arguments.llm_model,
        representative_papers=arguments.representatives,
        timeout_seconds=arguments.llm_timeout,
        context_window_tokens=arguments.llm_context,
        refresh=arguments.refresh,
    )
    print(f"LLM labels generated: {result.generated}")
    print(f"Title-based labels: {result.title_based}")
    print(f"Labels reused: {result.reused}")


if __name__ == "__main__":
    main()
