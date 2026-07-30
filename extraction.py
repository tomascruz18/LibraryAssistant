"""Extract and clean text from PDFs, with optional OCR fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import subprocess
from uuid import uuid4

import fitz


DEFAULT_MINIMUM_NATIVE_CHARACTERS = 100


@dataclass(frozen=True)
class ExtractionResult:
    text: str
    pages: list[str]
    method: str
    source_pdf: Path
    ocr_pdf: Path | None = None


def clean_text(text: str) -> str:
    """Remove common PDF line-break artifacts."""
    text = re.sub(r"-\s*\n\s*", "", text)
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def extract_pages(pdf_path: str | Path) -> list[str]:
    """Extract text from each page of a PDF using PyMuPDF."""
    path = Path(pdf_path)
    with fitz.open(path) as document:
        return [page.get_text() for page in document]


def _run_ocr(source: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"{source.stem}.ocr.pdf"
    temporary_path = output_dir / f".{source.stem}.{uuid4().hex}.tmp.pdf"

    try:
        result = subprocess.run(
            [
                "ocrmypdf",
                "--skip-text",
                "--deskew",
                "--rotate-pages",
                str(source),
                str(temporary_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not temporary_path.exists():
            message = result.stderr.strip() or "OCRmyPDF did not produce an output file."
            raise RuntimeError(message)
        os.replace(temporary_path, final_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return final_path


def pdf_to_text(
    pdf_path: str | Path,
    *,
    method: str = "auto",
    ocr_output_dir: str | Path = "data/ocr",
    minimum_native_characters: int = DEFAULT_MINIMUM_NATIVE_CHARACTERS,
) -> ExtractionResult:
    """Extract a PDF with ``native``, ``ocr``, or automatic selection."""
    source = Path(pdf_path)
    if method not in {"auto", "native", "ocr"}:
        raise ValueError("method must be 'auto', 'native', or 'ocr'")

    pages = extract_pages(source)
    native_text = clean_text("\n".join(pages))
    should_ocr = method == "ocr" or (
        method == "auto" and len(native_text) < minimum_native_characters
    )

    if not should_ocr:
        return ExtractionResult(native_text, pages, "native", source)

    ocr_pdf = _run_ocr(source, Path(ocr_output_dir))
    ocr_pages = extract_pages(ocr_pdf)
    return ExtractionResult(
        clean_text("\n".join(ocr_pages)),
        ocr_pages,
        "ocr",
        source,
        ocr_pdf,
    )
