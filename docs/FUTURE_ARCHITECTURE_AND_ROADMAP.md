# LibraryAssistant Architecture and Roadmap

> This document captures a possible future architecture. The initial prototype files
> discussed in the audit have since been moved unchanged to `scripts/`, and the project
> currently follows the simpler flat MVP structure in the main README.

## 1. Executive summary

LibraryAssistant is currently an exploratory prototype. The experiments demonstrate that
Zotero access, PDF extraction, embeddings, BM25, a similarity graph, Leiden clustering,
UMAP, and Plotly can be combined. They do not yet form a reproducible application:
the code has no stable document identifiers, persistent catalog, artifact metadata,
configuration layer, tests, or boundary between library code and executable examples.

The first product milestone should therefore be a trustworthy local search engine, not
an LLM interface. Its defining property is that every result can be traced to a Zotero
item, attachment, page, extracted-text version, and index version. Graph analysis and
LLM answers should be built only after that foundation is measurable and repeatable.

## 2. Audit of the current repository

### What is already useful

- `scripts/zotero_library.py` proves local Zotero API access and basic metadata retrieval.
- `scripts/advanced_text_extraction.py` proves native PDF extraction with an OCR fallback.
- `scripts/embedding_example.py` proves BGE-M3 embedding and cosine-similarity calls.
- `scripts/graph_example.py` is a useful end-to-end research spike covering paper-level
  embeddings, BM25, hybrid scoring, nearest neighbours, a k-nearest-neighbour graph,
  Leiden clustering, UMAP, and two plotting approaches.
- `scripts/app.py` proves the Streamlit entry point can remain very small.
- The README gives a coherent product direction and correctly emphasizes local-first,
  modular operation.

### Correctness and data-integrity risks

1. **Identity is positional.** Papers, embeddings, graph vertices, and search results are
   joined by list/array position. A changed Zotero query or filtering order can associate
   an embedding with the wrong paper. Use a stable internal `document_id`, retain the
   Zotero item key, and persist explicit ID-to-row mappings.
2. **`embeddings.npy` has no manifest.** The code does not verify its row count, source
   texts, model/revision, normalization setting, dimensions, or creation time.
3. **The graph can contain duplicate parallel edges.** The current undirected graph adds
   both `(i, j)` and `(j, i)` when each item selects the other as a neighbour. This can
   distort edge weights and Leiden communities. Define a symmetrization rule and store
   one canonical edge per pair.
4. **The README overstates implementation status.** Semantic search, BM25, UMAP, and
   Leiden exist inside a single experiment, but are not reusable application features.
   Full-library PDF ingestion, OCR management, hybrid reranking, LLM workflows, and a
   functional UI are not implemented.
5. **Retrieval is paper-level and abstract-only.** It cannot return page-level evidence,
   search papers without abstracts, or support grounded answers. Full text should be
   chunked with page and character provenance.
6. **OCR writes beside the source PDF.** The `_ocr.pdf` output is placed in the Zotero
   storage directory. Generated files should never modify or clutter the source library;
   use an application-owned cache and atomic writes.
7. **Metadata extraction is incomplete.** Zotero creators are structured records, not an
   `author` field. Dates are stored as unparsed strings, attachments and parent items are
   not joined, and collections/tags/notes are omitted.
8. **Hybrid score calibration is ad hoc.** Per-query min-max normalization and
   `alpha=0.9` have no evaluation basis and can behave poorly for constant or outlier
   score distributions. Reciprocal-rank fusion is a robust first baseline; learned or
   weighted fusion should follow an evaluation set.
9. **Dense similarity is quadratic.** A full cosine matrix is acceptable for a few
   hundred papers but does not scale. Retrieval and graph construction should use a
   nearest-neighbour index and batched processing.
10. **Importing modules performs work.** Most scripts connect to Zotero, load models,
    print data, show figures, or enter an input loop at module import time. Library
    modules must be side-effect free; executable behavior belongs behind CLI commands or
    `if __name__ == "__main__":`.

### Engineering and operational gaps

- Absolute, user-specific Windows paths and magic values (`292`, thresholds, model name)
  are embedded in source.
- There is no package layout, typed domain model, configuration, logging, error policy,
  migration strategy, test suite, or CI.
- PDF handles are not explicitly scoped with context managers, OCR tool availability is
  not checked, and OCR outputs have no cleanup or collision policy.
- Dependency versions and supported Python versions are not declared. The checked local
  virtual environment also fails to start because its standard-library path is broken;
  it should be recreated before development continues.
- Large models may download on first use, so “local-first” needs an explicit model setup,
  offline behavior, and storage policy.
- No search-quality corpus, latency budget, or regression measurements exist.
- No generated-data directory or ignore/backup policy exists.
- The current UI is only a title and “Hello world”; no application services are wired to
  it.

## 3. Architectural principles

1. **Zotero is read-only source data.** Never write generated artifacts into Zotero
   storage.
2. **Stable IDs, explicit mappings.** Arrays are optimization details, never identities.
3. **Immutable derived artifacts.** A content or configuration change produces a new
   version; a manifest records exactly how it was built.
4. **Incremental by default.** Reprocess only new, changed, or deleted items.
5. **Evidence before generation.** Retrieval returns page-addressable chunks before an
   LLM is allowed to answer.
6. **Core logic is UI-agnostic.** CLI, tests, and Streamlit call the same services.
7. **Measure before tuning.** Keep a small labelled query set and evaluate every
   retrieval or chunking change.

## 4. Proposed code structure

```text
LibraryAssistant/
├── pyproject.toml
├── README.md
├── .env.example
├── src/
│   └── library_assistant/
│       ├── __init__.py
│       ├── config.py                 # validated settings and paths
│       ├── domain/
│       │   ├── models.py             # Document, Attachment, Chunk, SearchHit
│       │   └── identifiers.py        # stable IDs and content hashes
│       ├── adapters/
│       │   ├── zotero.py             # read-only Zotero gateway
│       │   ├── pdf.py                # PyMuPDF extraction
│       │   ├── ocr.py                # OCRmyPDF subprocess adapter
│       │   └── ollama.py             # local generation adapter
│       ├── ingestion/
│       │   ├── catalog.py            # metadata/attachment synchronization
│       │   ├── extraction.py         # page text and extraction diagnostics
│       │   └── chunking.py            # provenance-preserving chunks
│       ├── indexing/
│       │   ├── semantic.py
│       │   ├── lexical.py
│       │   └── manifests.py
│       ├── retrieval/
│       │   ├── semantic.py
│       │   ├── lexical.py
│       │   ├── fusion.py
│       │   └── filters.py
│       ├── analysis/
│       │   ├── graph.py
│       │   ├── clustering.py
│       │   └── projection.py
│       ├── generation/
│       │   ├── prompts.py
│       │   ├── citations.py
│       │   └── qa.py
│       ├── storage/
│       │   ├── catalog.py             # SQLite repositories/migrations
│       │   └── artifacts.py           # file-backed artifact store
│       ├── services/
│       │   ├── sync_library.py
│       │   ├── build_indexes.py
│       │   ├── search_library.py
│       │   └── answer_question.py
│       ├── cli.py
│       └── ui/
│           └── streamlit_app.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── scripts/                           # one-off migration/benchmark tools
├── examples/                          # small, non-production demonstrations
├── docs/
└── var/                               # generated locally; ignored by Git
```

Use `src/` packaging to prevent accidental imports from the working directory. Keep
framework code at the edge: Streamlit should render service results, not contain
retrieval logic. Move the current experiments to `examples/` only after their reusable
parts have tests in the package.

## 5. Proposed data model

### Canonical catalog (SQLite)

SQLite is the source of truth for identifiers, metadata, provenance, and job state. It
supports transactions and migrations while keeping deployment local and simple.

| Entity | Important fields |
|---|---|
| `documents` | `document_id` (UUID), `zotero_item_key` (unique), `item_type`, `title`, `abstract`, `date_raw`, `year`, `doi`, `url`, `metadata_hash`, `deleted_at` |
| `creators` | `creator_id`, normalized name fields, optional ORCID |
| `document_creators` | `document_id`, `creator_id`, `creator_type`, `position` |
| `attachments` | `attachment_id`, `document_id`, `zotero_item_key`, source path, MIME type, size, mtime, SHA-256, availability |
| `collections` / `tags` | stable source keys/names plus many-to-many join tables |
| `extractions` | `extraction_id`, `attachment_id`, source hash, method, tool version, status, page count, quality metrics, artifact path |
| `chunks` | `chunk_id`, `extraction_id`, ordinal, page range, character offsets, token count, text hash, text artifact reference |
| `index_versions` | index ID, kind, model/revision, dimensions, parameters, input fingerprint, status, artifact path |
| `runs` | run ID, command, configuration hash, start/end times, status, counts, error summary |

Important constraints:

- `zotero_item_key` is the external identity; `document_id` is the internal identity.
- `chunk_id` should be deterministic from extraction identity, chunking version, and
  ordinal or text hash.
- Source hashes decide whether extraction and indexing can be reused.
- Deletion is first recorded as a tombstone so stale index rows can be detected and
  removed safely.

### Generated artifacts

```text
var/
├── catalog.sqlite
├── artifacts/
│   ├── text/<attachment_id>/<source_sha256>/pages.jsonl.zst
│   ├── chunks/<chunking_fingerprint>/chunks.parquet
│   ├── embeddings/<index_id>/
│   │   ├── vectors.npy
│   │   ├── rows.parquet             # row_number -> chunk_id/document_id
│   │   └── manifest.json
│   ├── lexical/<index_id>/
│   │   ├── index/...
│   │   └── manifest.json
│   ├── graphs/<graph_id>/
│   │   ├── nodes.parquet
│   │   ├── edges.parquet
│   │   └── manifest.json
│   └── projections/<projection_id>/
│       ├── coordinates.parquet
│       └── manifest.json
├── cache/
│   ├── ocr/
│   └── models/
├── logs/
└── exports/                           # user-requested, reproducible outputs
```

`pages.jsonl.zst` should contain one record per page with page number, extracted text,
method, and optional quality diagnostics. Parquet is appropriate for ID mappings,
chunks, graph tables, and projection coordinates; NumPy or a vector-index-native format
is appropriate for dense vectors.

Every `manifest.json` should include:

```json
{
  "schema_version": 1,
  "artifact_id": "immutable-id",
  "artifact_type": "semantic_index",
  "created_at": "ISO-8601 UTC timestamp",
  "code_version": "Git commit or package version",
  "input_fingerprint": "sha256:...",
  "record_count": 0,
  "model": {"name": "BAAI/bge-m3", "revision": "pinned-revision"},
  "parameters": {"normalize_embeddings": true},
  "files": [{"path": "vectors.npy", "sha256": "..."}]
}
```

This manifest makes cache invalidation and mismatch checks deterministic. Artifact paths
in SQLite should always be relative to the configured application data root so the
library can be moved or backed up.

### Retrieval and graph records

A search hit should have one stable contract across CLI and UI:

```text
SearchHit
  chunk_id, document_id, title
  page_start, page_end, snippet
  semantic_score, lexical_score, fused_score, rank
  index_versions, source_attachment_id
```

Graph nodes should reference `document_id`, not embedding row numbers. Graph edges should
store canonical `source_document_id < target_document_id`, similarity, construction
method, and graph version. Decide and record whether mutual-kNN or union-kNN is used;
deduplicate before clustering. UMAP coordinates are a replaceable view of a graph/index,
not canonical document data.

## 6. Target processing flow

```text
Zotero metadata + attachment references
        ↓ incremental sync (keys, versions, hashes)
SQLite catalog
        ↓ native extraction; OCR only when quality rules require it
Versioned page text
        ↓ provenance-preserving chunking
Versioned chunks
        ├── semantic index
        └── lexical index
                ↓ rank fusion + metadata filters
          page-addressable SearchHit records
                ├── graph/projection analysis
                └── grounded LLM answer with citations
```

## 7. Delivery roadmap

### Phase 0 — Reproducible foundation

- Add `pyproject.toml`, a supported Python version, locked dependency workflow, `src/`
  package, configuration, logging, and test setup.
- Recreate the broken virtual environment and document native dependencies
  (Tesseract/OCRmyPDF/Ollama).
- Add `.gitignore` rules for `var/`, model caches, generated figures, and local secrets.
- Convert examples so imports have no side effects.

**Exit criteria:** a clean environment can install the package; lint/unit-test commands
run; configuration is validated; no source file contains a user-specific absolute path.

### Phase 1 — Trustworthy catalog and extraction

- Implement read-only Zotero synchronization for items, creators, tags, collections,
  notes, and PDF attachments.
- Add SQLite migrations, stable IDs, content hashes, tombstones, and run records.
- Extract per-page text into the application artifact store; add OCR detection, tool
  checks, atomic output, quality metrics, and failure recovery.
- Implement deterministic, page-aware chunks.

**Exit criteria:** rerunning an unchanged sync does no extraction work; changing one PDF
invalidates only its descendants; every chunk resolves to an existing Zotero item,
attachment hash, and page range; Zotero storage remains unchanged.

### Phase 2 — Search minimum viable product

- Build versioned semantic and lexical indexes over chunks with explicit row mappings.
- Implement semantic, lexical, and reciprocal-rank-fusion retrieval plus filters for
  author, year, collection, tag, and item type.
- Create a small labelled evaluation set from real research questions.
- Add a CLI for `sync`, `index`, `search`, `doctor`, and `status`.

**Exit criteria:** no stale/mismatched index can load; top-k results include page-level
snippets and provenance; retrieval quality and latency are reported on the evaluation
set; a paper without a Zotero abstract remains searchable through its PDF.

### Phase 3 — Similarity graph and exploration

- Aggregate chunk evidence into a documented paper-level representation.
- Build a deduplicated mutual- or union-kNN graph without a full dense similarity matrix.
- Version graph parameters, Leiden seed/settings, and UMAP seed/settings.
- Add cluster summaries and interactive Plotly selection linked to search results.

**Exit criteria:** graph construction is deterministic for fixed inputs/settings; every
node and edge maps to stable document IDs; isolated papers remain visible; the UI can
filter and inspect clusters without recomputing the graph.

### Phase 4 — Grounded assistant and Streamlit UI

- Implement search, paper detail, related-paper, graph, and processing-status views.
- Add Ollama-based summarization, question answering, and comparison using retrieved
  chunks only.
- Render citations that open the correct local attachment/page where supported.
- Add prompt-injection boundaries, context-size limits, cancellation, and answer/run
  provenance.

**Exit criteria:** factual answer claims link to retrieved evidence; model and prompt
versions are recorded; unavailable evidence produces an explicit “not found” response;
the UI does not directly access storage internals.

### Phase 5 — Reliability and expansion

- Background job queue, resumable processing, progress reporting, backups, and
  diagnostics.
- Incremental citation graph and support for notes, reports, presentations, and code
  through type-specific extractors.
- Export/import of reproducible searches, reviews, and collections.
- Performance tests at expected library size and documented privacy/offline guarantees.

**Exit criteria:** interrupted jobs resume safely; catalog plus artifacts can be backed up
and restored; schema/artifact migrations are tested; supported collection sizes meet
documented latency and disk budgets.

## 8. Near-term implementation order

The next small vertical slice should be:

1. Package/config/test foundation.
2. Sync 10 Zotero papers and their PDF attachments into SQLite.
3. Extract and chunk those PDFs with full provenance.
4. Build both indexes and return a unified `SearchHit`.
5. Expose the same search through a CLI and a minimal Streamlit page.
6. Add graph analysis only after the index mappings and evaluation harness are stable.

This order proves the most important data contracts early and prevents the current
experiments from hardening into incompatible subsystems.

## 9. Decisions to make before implementation

- Expected library size and acceptable indexing time/disk usage.
- Whether application data should default to the repository, an OS-specific user-data
  directory, or a user-selected location.
- Whether PDF files are referenced in place or optionally copied into a managed store.
- Required offline behavior and how embedding/LLM model revisions are provisioned.
- Preferred vector backend after measuring exact search at the expected scale.
- Whether citations must deep-link into Zotero, an embedded PDF viewer, or both.

These choices affect adapters and storage configuration, but not the core identities and
artifact contracts proposed above.
