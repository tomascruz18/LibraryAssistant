# Persistent Embeddings

Embeddings are generated from the effective metadata stored in `data/library.sqlite3`.
The current MVP representation contains only the fields requested for retrieval:

```text
Authors: author one; author two

Date: 2006

Abstract: paper abstract or LLM-generated summary
```

Titles are deliberately not part of this first representation. The text format has an
internal version, so changing its composition later can deliberately refresh vectors.

## Build or update the index

After saving papers to the SQLite catalog, run this from the project root:

```powershell
python scripts\build_embeddings.py
```

The first run downloads/loads `BAAI/bge-m3` and embeds every eligible stored paper. A
later run embeds only papers whose authors, date, or abstract changed. It reports how
many vectors were generated and reused.

Use a different database, model, or batch size if needed:

```powershell
python scripts\build_embeddings.py --database data\library.sqlite3 --model BAAI/bge-m3 --batch-size 8
```

Changing the model name intentionally regenerates all embeddings, because vectors from
different models must never be mixed.

## Storage and paper linkage

Vectors live in the SQLite `paper_embeddings` table, not in a separate positional
`.npy` file. Each row is keyed by the stable Zotero item key (`zotero_key`), with a
foreign-key reference to `papers.zotero_key`. The table also stores the model name, a
SHA-256 hash of the embedding input, vector dimensions, and the float32 vector bytes.

`embeddings.load_database_embedding_index()` joins vectors back to papers by Zotero key
and returns the two lists in one verified order for `search.semantic_search()`. This
prevents a changed sort order from associating a vector with the wrong paper.

Unsupported or deleted papers have any existing vector removed. A paper with no
authors, date, or abstract is skipped.
