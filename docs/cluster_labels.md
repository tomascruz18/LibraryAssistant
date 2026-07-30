# Cluster Names

After graph construction and Leiden clustering, run:

```powershell
python scripts\label_clusters.py
```

For each multi-paper cluster, the tool selects up to five papers with the highest
weighted graph degree. It supplies their titles and short abstract excerpts to the local
LLM, which returns a concise research-theme name and one-sentence description. A
one-paper cluster uses a shortened form of its paper title instead, without calling the
LLM.

Labels are saved in `cluster_labels` with the embedding model, graph fingerprint, cluster
ID, label, description, source, LLM model, and labeling-pipeline version. A label is
reused only when the current graph fingerprint matches. If the graph changes, cluster IDs
may have a different meaning, so new label proposals are generated instead of silently
reusing old names.

Use `--refresh` to regenerate automatic labels for the current graph:

```powershell
python scripts\label_clusters.py --refresh
```

Manual labels are supported by the storage schema (`source = manual`) and are preserved
when automatic labels are refreshed. The current query tool shows labels beside cluster
sizes:

```powershell
python scripts\query_library.py --clusters
```
