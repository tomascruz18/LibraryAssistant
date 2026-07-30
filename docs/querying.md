# Querying the Stored Library

The query tool uses the saved SQLite catalog and persistent embeddings; it does not
reload Zotero or recreate vectors.

```powershell
python scripts\query_library.py
```

Interactive commands:

| Command | Action |
|---|---|
| `q cooling channels` | Hybrid semantic + BM25 search. |
| `s cooling channels` | Semantic-only search. |
| `b cooling channels` | BM25 keyword-only search. |
| `n 37DLSCKU` | Nearest papers from the full embedding similarity ranking. |
| `g 37DLSCKU` | Neighbours retained in the sparse saved graph. |
| `clusters` | List saved Leiden cluster sizes. |
| `c 3` | List papers in cluster 3. |
| `x` | Exit. |

Paper identity uses Zotero keys, not temporary numeric positions. One-shot commands are
also useful for testing:

```powershell
python scripts\query_library.py --query "regenerative cooling" --mode hybrid
python scripts\query_library.py --neighbors 37DLSCKU
python scripts\query_library.py --graph-neighbors 37DLSCKU
python scripts\query_library.py --clusters
```

Run `scripts\build_embeddings.py` first. The graph and cluster commands additionally
require `scripts\build_graph.py`.
