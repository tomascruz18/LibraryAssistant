"""Load paper metadata from a local Zotero library."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

from pyzotero import zotero

from extraction import pdf_to_text
from llm import (
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    METADATA_PIPELINE_VERSION,
    extract_metadata_with_retries,
    summarize_document_text,
)


def _creator_name(creator: dict[str, Any]) -> str:
    """Return a readable name from a Zotero creator record."""
    if creator.get("name"):
        return str(creator["name"])
    return " ".join(
        part
        for part in (creator.get("firstName", ""), creator.get("lastName", ""))
        if part
    )


def _missing_metadata_fields(paper: dict[str, Any]) -> set[str]:
    """Return the supported metadata fields that are absent from a paper record."""
    return {
        field
        for field in ("authors", "date", "abstract")
        if not paper.get(field)
    }


def _pdf_attachment(client: zotero.Zotero, item_key: str) -> dict[str, Any] | None:
    """Return identity/version information for the first PDF attachment."""
    for child in client.children(item_key):
        data = child.get("data", {})
        filename = str(data.get("filename", "")).lower()
        if data.get("contentType") == "application/pdf" or filename.endswith(".pdf"):
            return {
                "key": child.get("key") or data.get("key"),
                "version": child.get("version") or data.get("version") or 0,
                "filename": data.get("filename", ""),
            }
    return None


def _attachment_text(
    client: zotero.Zotero,
    attachment_key: str,
) -> tuple[str, str]:
    """Fetch a Zotero PDF and extract native text with automatic OCR fallback."""
    pdf_bytes = client.file(attachment_key)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporary_file:
        temporary_file.write(pdf_bytes)
        temporary_path = Path(temporary_file.name)

    try:
        extraction = pdf_to_text(
            temporary_path,
            method="auto",
            ocr_output_dir=Path("data") / "ocr",
        )
    finally:
        temporary_path.unlink(missing_ok=True)

    return extraction.text, extraction.method


def _enrich_missing_metadata(
    client: zotero.Zotero,
    paper: dict[str, Any],
    *,
    model: str,
    max_characters: int,
    timeout_seconds: float,
    context_window_tokens: int,
) -> None:
    """Fill only absent metadata values on an in-memory paper record."""
    missing_fields = _missing_metadata_fields(paper)
    if not missing_fields:
        return

    item_key = paper["id"]
    attachment = _pdf_attachment(client, item_key)
    if not attachment:
        paper["metadata_error"] = "No PDF attachment found for this Zotero item."
        return
    attachment_key = attachment["key"]
    paper["metadata_attachment_key"] = attachment_key
    paper["metadata_attachment_version"] = attachment["version"]

    try:
        document_text, extraction_method = _attachment_text(client, attachment_key)
        paper["text_extraction_method"] = extraction_method
        generated, attempt_count, _raw_response = extract_metadata_with_retries(
            document_text[:max_characters],
            model=model,
            timeout_seconds=timeout_seconds,
            context_window_tokens=context_window_tokens,
        )
        paper["metadata_llm_attempts"] = attempt_count
    except Exception as error:
        paper["metadata_error"] = str(error)
        return

    paper["document_type"] = generated["document_type"]
    if generated["document_type"] != "research_paper":
        paper["is_supported"] = False
        paper["metadata_error"] = (
            "Skipped unsupported document type: "
            f"{generated['document_type']}."
        )
        return

    for field in missing_fields:
        generated_value = generated[field]
        if generated_value:
            paper[field] = generated_value
            paper["metadata_source"][field] = "llm_extracted"

    # A missing explicit abstract is different from a failed metadata extraction. In
    # that case, produce a clearly labelled summary from the entire document.
    if "abstract" in missing_fields and not paper["abstract"]:
        try:
            paper["abstract"] = summarize_document_text(
                document_text,
                model=model,
                timeout_seconds=timeout_seconds,
                context_window_tokens=context_window_tokens,
            )
            paper["metadata_source"]["abstract"] = "llm_summary"
        except Exception as error:
            paper["metadata_error"] = str(error)
            return


def load_zotero_library(
    library_id: str = "0",
    library_type: str = "user",
    *,
    local: bool = True,
    limit: int | None = None,
    require_abstract: bool = False,
    enrich_missing_metadata: bool = True,
    llm_model: str = DEFAULT_MODEL,
    max_metadata_characters: int = 12_000,
    llm_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    llm_context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
    include_unsupported_documents: bool = False,
) -> list[dict[str, Any]]:
    """Return normalized paper records from Zotero.

    The local Zotero client must be running when ``local=True``. Attachments, notes,
    and annotations are omitted because they are not standalone papers. When
    ``enrich_missing_metadata`` is true, missing authors and dates are extracted from
    the opening PDF text. A missing abstract is first extracted only when an explicit
    abstract section exists; otherwise an abstract-style summary is generated from the
    full document in context-safe chunks. Presentations, lectures, and other
    non-paper documents detected during LLM extraction are excluded by default. Set
    ``include_unsupported_documents=True`` to inspect them. Zotero is never modified;
    enrichment exists only in the returned records.
    """
    client = zotero.Zotero(
        library_id=library_id,
        library_type=library_type,
        local=local,
    )
    # For an MVP preview, ask Zotero for only the requested top-level items instead
    # of downloading the complete library and filtering afterwards.
    items = (
        client.top(limit=limit)
        if limit is not None
        else client.everything(client.top())
    )
    papers: list[dict[str, Any]] = []

    for item in items:
        item_data = item.get("data", {})
        if item_data.get("itemType") in {"attachment", "note", "annotation"}:
            continue

        creators = [
            name
            for creator in item_data.get("creators", [])
            if (name := _creator_name(creator))
        ]
        paper = {
            "id": item.get("key") or item_data.get("key"),
            "zotero_version": item.get("version") or item_data.get("version") or 0,
            "title": str(item_data.get("title", "")).strip(),
            "abstract": str(item_data.get("abstractNote", "")).strip(),
            "date": str(item_data.get("date", "")).strip(),
            "authors": creators,
            "item_type": item_data.get("itemType", ""),
            "doi": str(item_data.get("DOI", "")).strip(),
            "url": str(item_data.get("url", "")).strip(),
            "metadata_source": {
                "authors": "zotero" if creators else None,
                "date": "zotero" if item_data.get("date", "").strip() else None,
                "abstract": "zotero" if item_data.get("abstractNote", "").strip() else None,
            },
            "zotero_metadata": {
                "authors": creators,
                "date": str(item_data.get("date", "")).strip(),
                "abstract": str(item_data.get("abstractNote", "")).strip(),
            },
            "metadata_pipeline_version": METADATA_PIPELINE_VERSION,
            "llm_model": llm_model,
            "llm_context_window_tokens": llm_context_window_tokens,
            "document_type": "unknown",
            "is_supported": True,
        }
        if enrich_missing_metadata:
            _enrich_missing_metadata(
                client,
                paper,
                model=llm_model,
                max_characters=max_metadata_characters,
                timeout_seconds=llm_timeout_seconds,
                context_window_tokens=llm_context_window_tokens,
            )
        if not paper["is_supported"] and not include_unsupported_documents:
            continue
        if require_abstract and not paper["abstract"]:
            continue

        papers.append(paper)

        if limit is not None and len(papers) >= limit:
            break

    return papers
