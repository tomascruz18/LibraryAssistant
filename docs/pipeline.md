# End-to-End Pipeline

The complete MVP can be generated with one command:

```powershell
python scripts\build_all.py --papers 1000
```

It runs, in order:

1. Zotero loading plus missing-metadata extraction/summarization;
2. SQLite paper storage;
3. incremental embeddings;
4. the 90th-percentile/top-k similarity graph and Leiden clustering;
5. local-LLM names for multi-paper clusters;
6. cached UMAP coordinates and both interactive HTML visualizations.

The normal command updates the requested Zotero entries and reuses downstream results
whose inputs did not change. Existing database records outside the requested Zotero limit
remain available.

## Full rebuild

To regenerate the selected library from scratch:

```powershell
python scripts\build_all.py --papers 1000 --rebuild
```

The rebuild is created in a temporary SQLite database. The existing catalog is replaced
only after every pipeline stage succeeds, so an embedding, graph, or visualization error
does not destroy the working database.

If paper metadata already exists and only downstream work is missing, skip Zotero and the
LLM entirely:

```powershell
python scripts\build_all.py --database-only
```

## Main settings

```powershell
python scripts\build_all.py `
  --papers 1000 `
  --max-paper-tokens 24000 `
  --similarity-percentile 90 `
  --neighbors 10
```

| Option | Default | Meaning |
|---|---:|---|
| `--papers` | 1000 | Maximum number of top-level Zotero entries requested. |
| `--max-paper-tokens` | 24000 | Skip fallback summaries above this estimated size. |
| `--similarity-percentile` | 90 | Global graph-similarity percentile cutoff. |
| `--neighbors` | 10 | Maximum graph neighbours retained per paper. |
| `--cluster-label-representatives` | 5 | Central papers provided to cluster naming. |
| `--front-matter-characters` | 12000 | PDF opening text sent to metadata extraction. |
| `--llm-context` | 8000 | Ollama context-window size. |
| `--llm-timeout` | 120 | Timeout in seconds for each Ollama request. |
| `--embedding-batch-size` | 16 | Sentence Transformer batch size. |
| `--umap-neighbors` | 15 | UMAP neighbourhood size. |
| `--umap-min-dist` | 0.1 | UMAP minimum-distance parameter. |

Model names and output paths can also be changed with `--llm-model`,
`--embedding-model`, `--umap-output`, and `--force-output`. Run
`python scripts\build_all.py --help` for the complete option list.

Use `--skip-cluster-labels` to omit the LLM naming stage. Single-paper clusters use a
shortened paper title and do not require an LLM call.
