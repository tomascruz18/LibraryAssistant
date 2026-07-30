# UMAP Paper Map

The visualization converts the stored paper embeddings into a two-dimensional UMAP map,
colours papers by their saved Leiden cluster, and draws the persisted similarity edges.

Build the embeddings and graph first, then run:

```powershell
python scripts\build_visualization.py
```

This writes two interactive, self-contained Plotly HTML files:

- `data/library_map.html`: the UMAP semantic-similarity map;
- `data/library_force_graph.html`: a Fruchterman-Reingold force layout of the stored
  similarity graph.

Open either file in a browser and hover over a point to see its title. Papers are
coloured with light, evenly distributed colours by Leiden cluster. The force layout uses
the stored graph topology and similarity weights, while UMAP uses the full embeddings.

## Cached coordinates

UMAP coordinates are saved in `paper_projections`, keyed by Zotero key and embedding
model. `projection_state` stores only the embedding fingerprint and UMAP settings
(`n_neighbors=15`, `min_dist=0.1`, `random_state=42`). Re-running the command reuses
coordinates unless the embeddings or one of those settings changed.

The coordinates are a visualization cache, not scientific metadata. They may change when
the embedding model or library changes; the stable records remain the papers, embeddings,
graph edges, and cluster assignments.
