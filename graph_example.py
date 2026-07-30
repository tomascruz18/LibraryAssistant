from pyzotero import zotero
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rank_bm25 import BM25Okapi
import re
import plotly.express as px
import pandas as pd

zot = zotero.Zotero(
    library_id="0",
    library_type="user",
    local=True
)

items = zot.everything(
    zot.items()
)

print(len(items))

for item in items[:10]:
    print(item["data"]["title"])

papers = []

for item in items:

    data = item["data"]

    abstract = data.get("abstractNote", "")

    if abstract.strip():

        papers.append({
            "title": data.get("title", ""),
            "abstract": abstract,
            "year": data.get("date", ""),
            "author": data.get("author", "")
        })

    if len(papers) == 292:
        break

for i, paper in enumerate(papers):

    print("="*80)
    print(f"[{i}]")
    print(paper["title"])
    print()
    print(paper["abstract"][:500])


model = SentenceTransformer("BAAI/bge-m3")  # allenai/specter2 potentially better

texts = []

for paper in papers:

    texts.append(
        paper["title"] +
        "\n\n" +
        paper["abstract"]
    )

STOPWORDS = {
    "a", "an", "the", "of", "to", "and", "for",
    "in", "on", "with", "by", "ii", "iii", "iv"
}

def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    return [
        w
        for w in words
        if len(w) >= 3 and w not in STOPWORDS
    ]

corpus = [
    tokenize(text)
    for text in texts
]

bm25 = BM25Okapi(corpus)

# embeddings = model.encode(
#     texts,
#     normalize_embeddings=True,
#     show_progress_bar=True
# )
# np.save("embeddings.npy", embeddings)
embeddings = np.load("embeddings.npy")

print(embeddings.shape)
similarity = cosine_similarity(embeddings)

print(similarity)

values = []

for i in range(len(similarity)):
    for j in range(i + 1, len(similarity)):
        values.append(similarity[i, j])

print(min(values))
print(max(values))
print(np.mean(values))
print(np.percentile(values, [50, 75, 90, 95, 99]))


k = 10
threshold = 0.55
edges = []
for i in range(len(embeddings)):
    idx = np.argsort(similarity[i])[::-1]
    count = 0

    for j in idx[1:]:

        if similarity[i,j] < threshold:
            break
        edges.append((i,j,similarity[i,j]))
        count += 1
        if count == k:
            break


import igraph as ig

g = ig.Graph()

g.add_vertices(len(embeddings))

# Add the edges
g.add_edges([(i, j) for i, j, _ in edges])

# Add the corresponding weights
g.es["weight"] = [w for _, _, w in edges]

import leidenalg

clusters = leidenalg.find_partition(
    g,
    leidenalg.ModularityVertexPartition,
    weights="weight"
)

print(clusters)
print(clusters.membership)
membership = clusters.membership

for cluster_id, nodes in enumerate(clusters):

    print(f"\n=== Cluster {cluster_id} ===")

    for node in nodes:
        print(papers[node]["title"])

import umap

coords = umap.UMAP(random_state=42).fit_transform(
    embeddings
)

# from pyvis.network import Network
#
# net = Network(
#     height="900px",
#     width="100%",
#     bgcolor="white",
#     font_color="black"
# )
#
# for i, paper in enumerate(papers):
#     net.add_node(
#         i,
#         label=str(i),
#         title=paper["title"],
#         group=membership[i]
#     )
#
# for i, j, w in edges:
#     net.add_edge(
#         i,
#         j,
#         value=w
#     )
#
# net.write_html("papers.html", notebook=False)

df = pd.DataFrame({
    "x": coords[:, 0],
    "y": coords[:, 1],
    "cluster": membership,
    "title": [p["title"] for p in papers],
    "year": [p["year"] for p in papers],
    "id": list(range(len(papers)))
})

df["hover"] = [
    f"""
<b>{p['title']}</b><br>
Year: {p['year']}<br><br>
{p['abstract'][:250]}...
"""
    for p in papers
]

fig = px.scatter(
    df,
    x="x",
    y="y",
    color="cluster",
    hover_name="title",
    hover_data=None
)

fig.update_traces(
    hovertemplate="%{customdata[0]}",
    customdata=df[["hover"]]
)

fig.show()

import plotly.graph_objects as go

# -------------------------
# Build edge trace
# -------------------------
edge_x = []
edge_y = []

for i, j, w in edges:
    edge_x.extend([coords[i, 0], coords[j, 0], None])
    edge_y.extend([coords[i, 1], coords[j, 1], None])

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    mode="lines",
    line=dict(width=0.5, color="lightgray"),
    hoverinfo="none"
)

# -------------------------
# Build node trace
# -------------------------
node_trace = go.Scatter(
    x=coords[:, 0],
    y=coords[:, 1],
    mode="markers",
    hovertemplate="<b>%{text}</b><extra></extra>",
    text=[paper["title"] for paper in papers],
    marker=dict(
        size=10,
        color=membership,
        colorscale="Viridis",
        line=dict(width=0.5, color="black"),
        colorbar=dict(title="Cluster")
    )
)

# -------------------------
# Plot
# -------------------------
fig = go.Figure(data=[edge_trace, node_trace])

fig.update_layout(
    showlegend=False,
    hovermode="closest",
    template="plotly_white"
)

fig.show()


import matplotlib.pyplot as plt

plt.scatter(
    coords[:,0],
    coords[:,1],
    c=clusters.membership
)

plt.show()

def print_search_results(results):
    print()

    for i, r in enumerate(results):
        print(
            f"{i + 1:2d}. "
            f"[{r['similarity']:.3f}] "
            f"(emb={r['embedding']:.3f}, "
            f"bm25={r['bm25']:.3f}) "
            f"{r['title']}"
        )

    print()

def semantic_search(query, model, embeddings, papers, k=10):

    # Embed the query
    query_embedding = model.encode(["Represent this sentence for searching relevant passages: " + query], normalize_embeddings=True)

    # Compare against every paper
    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    # Sort descending
    order = np.argsort(similarities)[::-1]

    # Return the k best papers
    results = []

    for idx in order[:k]:
        results.append({
            "id": idx,
            "similarity": similarities[idx],
            "embedding": similarities[idx],
            "bm25": 0,
            "title": papers[idx]["title"],
            "year": papers[idx]["year"],
            "abstract": papers[idx]["abstract"][:250]
        })

    return results

results = semantic_search(
    "cooling channels",
    model,
    embeddings,
    papers
)

for r in results:
    print(f"{r['similarity']:.3f}  {r['title']}")

def bm25_search(query, bm25, papers, k=20):

    tokens = tokenize(query)

    scores = bm25.get_scores(tokens)

    order = np.argsort(scores)[::-1]

    results = []

    for idx in order[:k]:
        results.append({
            "id": idx,
            "similarity": scores[idx],      # BM25 score
            "embedding": 0,
            "bm25": scores[idx],
            "title": papers[idx]["title"],
            "year": papers[idx]["year"],
            "abstract": papers[idx]["abstract"][:250]
        })

    return results

results = bm25_search(
    "method of characteristics",
    bm25,
    papers
)

print_search_results(results)

def get_nearest_neighbors(paper_id, similarity_matrix, papers, k=10):
    """
    Returns the k papers most similar to the given paper.

    Parameters
    ----------
    paper_id : int
        Index of the reference paper.

    similarity_matrix : np.ndarray
        NxN cosine similarity matrix.

    papers : list
        List of paper dictionaries.

    k : int
        Number of neighbours to return.

    Returns
    -------
    list of dict
    """

    # Similarities of this paper to every other paper
    similarities = similarity_matrix[paper_id]

    # Sort from most similar to least similar
    order = np.argsort(similarities)[::-1]

    # Remove the paper itself (similarity = 1)
    order = order[order != paper_id]

    results = []

    for idx in order[:k]:
        results.append({
            "id": idx,
            "similarity": similarities[idx],
            "embedding": similarities[idx],
            "bm25": 0,
            "title": papers[idx]["title"],
            "year": papers[idx]["year"],
            "abstract": papers[idx]["abstract"],
        })

    return results

neighbors = get_nearest_neighbors(
    paper_id=0,
    similarity_matrix=similarity,
    papers=papers,
    k=10
)

def normalize(x):
    x = np.asarray(x)

    if np.max(x) == np.min(x):
        return np.zeros_like(x)

    return (x - np.min(x)) / (np.max(x) - np.min(x))

def hybrid_search(
    query,
    model,
    embeddings,
    bm25,
    papers,
    alpha=0.9,
    k=20
):

    query_embedding = model.encode(
        ["Represent this sentence for searching relevant passages: " + query],
        normalize_embeddings=True
    )

    embedding_scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    bm25_scores = bm25.get_scores(
        tokenize(query)
    )

    # Normalize
    embedding_scores = normalize(embedding_scores)
    bm25_scores = normalize(bm25_scores)

    scores = alpha * embedding_scores + (1-alpha) * bm25_scores

    order = np.argsort(scores)[::-1]

    results = []

    for idx in order[:k]:
        results.append({
            "id": idx,
            "similarity": scores[idx],
            "embedding": embedding_scores[idx],
            "bm25": bm25_scores[idx],
            "title": papers[idx]["title"],
            "year": papers[idx]["year"],
            "abstract": papers[idx]["abstract"][:250]
        })

    return results

print(f"Reference paper:")
print(papers[0]["title"])

print("\nNearest neighbours:\n")

for n in neighbors:
    print(f"{n['similarity']:.3f}  {n['title']}")


while True:

    command = input(
        "\nCommand (q <query>, n <paper_id>, x): "
    ).strip()

    if command.lower() == "x":
        break

    elif command.startswith("q "):

        query = command[2:].strip()

        results = hybrid_search(
            query=query,
            model=model,
            embeddings=embeddings,
            bm25=bm25,
            papers=papers,
            k=20
        )

        print(f"\nResults for '{query}'")
        print_search_results(results)

    elif command.startswith("n "):

        try:
            paper_id = int(command[2:].strip())

            print(f"\nReference paper:")
            print(f"[{paper_id}] {papers[paper_id]['title']}")
            print()

            neighbors = get_nearest_neighbors(
                paper_id=paper_id,
                similarity_matrix=similarity,
                papers=papers,
                k=10
            )

            print("Nearest neighbours")
            print_search_results(neighbors)

        except (ValueError, IndexError):
            print("Invalid paper id.")

    else:
        print("Unknown command.")

# unique_clusters = sorted(set(membership))
# cmap = plt.get_cmap("tab20")

# cluster_colors = {
#     cluster: cmap(i % 20)
#     for i, cluster in enumerate(unique_clusters)
# }
#
# vertex_colors = [
#     cluster_colors[c]
#     for c in membership
# ]
#
# fig, ax = plt.subplots(figsize=(5, 5))
# ig.plot(
#     g,
#     target=ax,
#     layout="fr",
#     vertex_size=30,
#     vertex_color=vertex_colors,
#     vertex_frame_width=4.0,
#     vertex_frame_color="white",
#     vertex_label_size=7.0,
#     edge_width = [5 * w for w in g.es["weight"]]
# )
#
# plt.show()

