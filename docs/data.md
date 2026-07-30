# Paper Data Model

This guide describes the in-memory paper records returned by `data.py` and the fields
that are later persisted by `storage.py`. For the SQLite file, inspection commands, and
incremental synchronization, see [Storage](storage.md).

## Data flow

```text
Zotero item + PDF attachment
        |
        +-- Existing Zotero metadata
        |
        +-- Missing metadata only
                |
                +-- Native PDF text or OCR fallback
                +-- LLM extraction or full-document summary
        |
        +-- Normalized paper record
```

`data.load_zotero_library()` does not modify Zotero. It produces normalized dictionaries
that can be searched in memory or saved to the local catalog.

## Paper record

Each record uses the Zotero item key as its stable identity:

```python
paper = {
    "id": "37DLSCKU",
    "zotero_version": 3978,
    "title": "...",
    "item_type": "conferencePaper",
    "authors": ["First Author", "Second Author"],
    "date": "2006",
    "abstract": "...",
    "doi": "...",
    "url": "...",
}
```

### Original Zotero metadata

`zotero_metadata` preserves the fields exactly as supplied by Zotero:

```python
paper["zotero_metadata"] = {
    "authors": [...],
    "date": "...",
    "abstract": "...",
}
```

The top-level `authors`, `date`, and `abstract` fields are the effective values that the
rest of LibraryAssistant should use. Keeping both allows a later Zotero correction to
replace an LLM-derived value without losing its history.

### Provenance

`metadata_source` records the origin of each effective value:

```python
paper["metadata_source"] = {
    "authors": "zotero" or "llm_extracted",
    "date": "zotero" or "llm_extracted",
    "abstract": "zotero", "llm_extracted", or "llm_summary",
}
```

- `llm_extracted` means the PDF contained an author-provided abstract, including an
  unlabeled abstract-like paragraph.
- `llm_summary` means no abstract was found and the LLM generated a summary from the
  complete document.

### PDF and document classification

When PDF processing is needed, these fields are added:

| Field | Meaning |
|---|---|
| `metadata_attachment_key` | Zotero key of the PDF used for processing. |
| `metadata_attachment_version` | Zotero version of that attachment. |
| `text_extraction_method` | `native` or `ocr`. |
| `document_type` | `research_paper`, `presentation_or_lecture`, `other`, or `unknown`. |
| `is_supported` | Whether the document belongs in the paper pipeline. |

Presentations, lectures, and other unsupported documents are excluded by default before
they reach the normal paper catalog.

### LLM configuration and status

```python
paper["metadata_pipeline_version"]
paper["llm_model"]
paper["llm_context_window_tokens"]
paper.get("metadata_llm_attempts")
paper.get("metadata_error")
```

The pipeline version and model settings identify how an LLM-derived value was created.
`metadata_llm_attempts` is present when metadata extraction ran; `1` means the first
LLM response passed validation, while `2` or `3` means an invalid or failed response was
retried successfully.
`metadata_error` is set when a PDF, OCR, or LLM call fails; no fabricated value is added.

## Processing states

Stored records are assigned one of these states:

- `complete`: authors, date, and abstract/summary are available.
- `partial`: at least one required field is still empty.
- `error`: processing failed and the saved error explains why.

## Loading examples

Load records without LLM/PDF enrichment:

```python
from data import load_zotero_library

papers = load_zotero_library(limit=20, enrich_missing_metadata=False)
```

Load with the normal enrichment behavior:

```python
papers = load_zotero_library(limit=20)
```

Use `include_unsupported_documents=True` only when deliberately inspecting detected
presentations or other non-paper documents.
