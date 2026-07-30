"""Test strict metadata extraction and context-safe summarization for one PDF.

Example:
    python scripts/llm_metadata_from_pdf.py "C:\\path\\to\\paper.pdf" --mode extract
    python scripts/llm_metadata_from_pdf.py "C:\\path\\to\\paper.pdf" --mode summarize
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from extraction import pdf_to_text
from llm import (
    DEFAULT_MODEL,
    MetadataExtractionError,
    extract_metadata_with_retries,
    summarize_document_text,
)


def extract_document_text(pdf_path: str | Path) -> tuple[str, str]:
    """Return full PDF text, using OCR automatically when native text is too short."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    result = pdf_to_text(
        path,
        method="auto",
        ocr_output_dir=PROJECT_ROOT / "data" / "ocr",
    )
    return result.text, result.method


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract metadata and, only when needed, summarize a PDF with Ollama."
    )
    parser.add_argument("pdf", type=Path, help="Path to the source PDF")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Installed Ollama model")
    parser.add_argument(
        "--mode",
        choices=("extract", "summarize", "auto"),
        default="auto",
        help=(
            "extract: strict front-matter extraction only; summarize: force a full "
            "document summary; auto: extract first, then summarize only if needed"
        ),
    )
    parser.add_argument(
        "--front-matter-characters",
        type=int,
        default=12_000,
        help="Characters sent to strict metadata extraction",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output file")
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="Save the unparsed Ollama metadata response for debugging",
    )
    arguments = parser.parse_args()

    if arguments.raw_output and arguments.mode == "summarize":
        parser.error("--raw-output is available only with --mode extract or --mode auto")

    document_text, extraction_method = extract_document_text(arguments.pdf)
    if arguments.mode == "summarize":
        metadata = {
            "authors": [],
            "date": None,
            "abstract": summarize_document_text(
                document_text,
                model=arguments.model,
            ),
            "abstract_source": "llm_summary",
            "document_type": "forced_summary",
        }
    else:
        try:
            metadata, attempt_count, raw_response = extract_metadata_with_retries(
                document_text[: arguments.front_matter_characters],
                model=arguments.model,
            )
        except MetadataExtractionError as error:
            if arguments.raw_output and error.raw_response is not None:
                arguments.raw_output.parent.mkdir(parents=True, exist_ok=True)
                arguments.raw_output.write_text(error.raw_response, encoding="utf-8")
            raise
        if arguments.raw_output:
            arguments.raw_output.parent.mkdir(parents=True, exist_ok=True)
            arguments.raw_output.write_text(raw_response, encoding="utf-8")
        metadata["metadata_llm_attempts"] = attempt_count

    metadata["text_extraction_method"] = extraction_method

    if arguments.mode == "extract":
        metadata["abstract_source"] = (
            "llm_extracted" if metadata["abstract"] else None
        )
    elif arguments.mode == "auto" and metadata["document_type"] != "research_paper":
        metadata["abstract"] = None
        metadata["abstract_source"] = None
        metadata["skip_reason"] = (
            "Unsupported document type: " f"{metadata['document_type']}"
        )
    elif arguments.mode == "auto" and metadata["abstract"]:
        metadata["abstract_source"] = "llm_extracted"
    elif arguments.mode == "auto":
        metadata["abstract"] = summarize_document_text(
            document_text,
            model=arguments.model,
        )
        metadata["abstract_source"] = "llm_summary"

    rendered = json.dumps(metadata, indent=2, ensure_ascii=False)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
