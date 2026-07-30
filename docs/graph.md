# Similarity Graph and Clusters

The graph is built from the persistent embeddings in `data/library.sqlite3`.

```powershell
python scripts\build_graph.py
```

The MVP uses a hybrid edge rule:

- calculate a global similarity cutoff at the **90th percentile** of unique paper pairs;
- for each paper, retain at most its **10** strongest neighbours;
- keep an edge only when it passes both rules.

This avoids a dense graph dominated by general papers while making the cutoff adapt to the
library's embedding distribution. Leiden clustering then assigns every paper to a
community, including isolated papers as one-paper communities.

## Stored data

Only the current graph state is stored:

- `paper_similarity_edges`: Zotero-key pair and cosine similarity;
- `paper_clusters`: Zotero key and Leiden cluster ID;
- `similarity_graph_state`: embedding model, input fingerprint, two graph parameters,
  calculated threshold, and build timestamp.

The graph is rebuilt only after the embedding input/model or graph configuration changes.
The cache therefore remains valid across application restarts and does not depend on a
vector-array position. The current graph and cluster assignments are replaced together;
we do not keep historical runs in this MVP.

To tune it later:

```powershell
python scripts\build_graph.py --percentile 95 --neighbors 15
```
