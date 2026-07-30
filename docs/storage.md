# SQLite Storage

LibraryAssistant stores processed paper metadata in a local SQLite database at
`data/library.sqlite3` by default. For a library of roughly 1,000 items, SQLite is small,
fast, transactional, and requires no separate database service.

For the fields in each paper record and how they are produced, see [Paper Data Model](data.md).

## Why not one JSON file per paper?

JSON files are useful for test outputs, but they make atomic updates, queries, change
detection, and schema evolution harder. SQLite still keeps the project local while
providing indexed lookup by Zotero key and safe updates if processing is interrupted.

## Database contents

The `papers` table uses the Zotero item key as its primary key. It stores:

- original Zotero metadata and effective enriched metadata;
- LLM provenance, document classification, and native/OCR extraction method;
- the number of LLM metadata-extraction attempts used for the stored result;
- Zotero item and attachment versions;
- pipeline/model settings, processing status, errors, and timestamps.

The `sync_state` table is a small key/value store for future state such as Zotero's last
library version or the timestamp of the last complete synchronization.

## Change detection

The catalog stores these change signals:

- `zotero_version` changes when the parent Zotero item changes.
- `attachment_key` and `attachment_version` identify the PDF used by the pipeline.
- `metadata_pipeline_version` identifies changes to extraction behavior.
- `source_fingerprint` hashes the relevant source inputs and configuration.

`storage.needs_processing()` compares a current Zotero record with the stored one. It
recognizes new, modified, restored, attachment-changed, pipeline-changed, and
model-configuration-changed items. Errors are retried only when
`retry_errors=True`, preventing a broken PDF from running on every startup.

## Saving and reading records

```python
from data import load_zotero_library
from storage import load_papers, save_papers

papers = load_zotero_library(limit=10)
saved_count = save_papers(papers)

stored_papers = load_papers()
print(saved_count, len(stored_papers))
```

Read one paper by Zotero key:

```python
from storage import get_paper

paper = get_paper("37DLSCKU")
print(paper["abstract"])
print(paper["metadata_source"])
```

## Inspecting the database

From the project root, use the read-only inspection script to see catalog health and
metadata provenance. It never modifies the database:

```powershell
python scripts\inspect_library_database.py
```

Useful views:

```powershell
python scripts\inspect_library_database.py --errors
python scripts\inspect_library_database.py --llm
python scripts\inspect_library_database.py --status complete --limit 50
python scripts\inspect_library_database.py --source zotero
```

| Command | Shows |
|---|---|
| no options | Database size, integrity, schema version, and aggregate counts |
| `--errors` | Failed records and their saved error messages |
| `--llm` | Papers whose abstract was extracted or summarized by the LLM |
| `--status complete` | Records with a chosen processing status (`complete`, `partial`, or `error`) |
| `--source zotero` | Records with a chosen abstract source (`zotero`, `llm_extracted`, `llm_summary`, or `missing`) |

Use `--limit 50` to expand a listing, or inspect another catalog path with:

```powershell
python scripts\inspect_library_database.py --database "path\to\library.sqlite3"
```

The default summary includes database size, integrity check, schema version, record
counts, processing statuses, abstract sources, PDF extraction methods, and document
types. The listing options print Zotero keys, titles, provenance, and any saved errors.

## Future incremental synchronization

The intended future loop is:

```text
Fetch lightweight Zotero item metadata
        |
        +-- key absent from SQLite ------------> new: process and insert
        +-- Zotero version changed ------------> modified: process and update
        +-- attachment version changed --------> PDF changed: process and update
        +-- LLM pipeline version changed ------> reprocess LLM-derived fields
        +-- none changed ----------------------> reuse stored record
```

`storage.needs_processing()` implements these comparisons. The current loader still
performs enrichment while loading; a later sync command should call the comparison
before downloading PDFs or invoking Ollama.

## Pipeline versioning

`llm.METADATA_PIPELINE_VERSION` identifies the extraction/summarization behavior. Bump
it only when a prompt or algorithm change should invalidate previously LLM-derived
metadata. Zotero-only metadata does not need reprocessing when this number changes.

## Privacy and Git

The `data/` directory is ignored except for `.gitkeep`. The SQLite catalog and test JSON
files can contain titles, authors, abstracts, local library identifiers, and generated
content, so they should remain local and be backed up separately if desired.
