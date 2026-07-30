"""Minimal Streamlit MVP for loading and searching Zotero abstracts."""

from __future__ import annotations

import streamlit as st

from data import load_zotero_library
from embeddings import generate_embeddings, load_embedding_model, paper_text
from search import build_bm25, hybrid_search


st.set_page_config(page_title="LibraryAssistant", layout="wide")
st.title("LibraryAssistant")
st.caption("Local hybrid search over titles and abstracts from Zotero.")

with st.sidebar:
    library_limit = st.number_input(
        "Maximum papers",
        min_value=10,
        max_value=10_000,
        value=300,
        step=10,
    )
    build_clicked = st.button("Load Zotero and build index", type="primary")

if build_clicked:
    with st.status("Building local search index...", expanded=True) as status:
        st.write("Loading papers from Zotero")
        papers = load_zotero_library(
            limit=int(library_limit),
            require_abstract=True,
        )
        if not papers:
            status.update(label="No papers with abstracts found", state="error")
            st.stop()

        texts = [paper_text(paper) for paper in papers]
        st.write(f"Embedding {len(papers)} papers")
        model = load_embedding_model()
        embeddings = generate_embeddings(texts, model)
        bm25 = build_bm25(texts)

        st.session_state["papers"] = papers
        st.session_state["model"] = model
        st.session_state["embeddings"] = embeddings
        st.session_state["bm25"] = bm25
        status.update(label=f"Indexed {len(papers)} papers", state="complete")

if "papers" not in st.session_state:
    st.info("Use the sidebar button to load the local Zotero library.")
    st.stop()

query = st.text_input("Search the library")
if query:
    results = hybrid_search(
        query,
        st.session_state["model"],
        st.session_state["embeddings"],
        st.session_state["bm25"],
        st.session_state["papers"],
        k=20,
    )
    for result in results:
        st.subheader(f"{result['rank']}. {result['title']}")
        author_text = ", ".join(result.get("authors", []))
        st.caption(
            f"{author_text} · {result.get('date', '')} · "
            f"score {result['score']:.3f}"
        )
        st.write(result.get("abstract", ""))
