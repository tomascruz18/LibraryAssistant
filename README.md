# LibraryAssistant
A local AI-powered literature assistant for exploring, searching and reasoning over a personal scientific paper library.

The project combines semantic search, keyword search, graph analysis and local large language models to create an interactive research environment.

## Features

- Import papers directly from a local Zotero library.
- Extract text and abstracts from PDFs. This can happen in two ways:
  - from zotero metadata
  - if metadata not available, info is created by a llm
- Generate embeddings using state-of-the-art embedding models.
- Perform semantic search over the entire library.
- Perform keyword search using BM25.
- Combine semantic and lexical search.
- Construct a similarity graph between papers.
- Detect research communities using the Leiden algorithm.
- Visualize the literature using UMAP.
- Query a local LLM (Ollama) about retrieved papers.
- Planned interactive web interface using Streamlit.

---

## Project Goals

The long-term vision is to build a personal research assistant capable of answering questions such as

> Which papers discuss regenerative cooling using conjugate heat transfer?

> Summarize the different approaches to multi-fidelity optimization.

> Compare the cooling correlations used in these five papers.

> Which papers are central to the regenerative cooling literature?

> Which papers are related to a specified paper? 

Unlike public tools, the assistant will also search personal notes, internal reports and unpublished documents.

---

## Current Pipeline

```
                Zotero Library
                       │
                       ▼
              Metadata + PDFs
                       │
                       ▼
             Text Extraction (PyMuPDF/OCR)
                       │
                       ▼
                 Embedding Model
                 (BAAI/bge-m3)
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
 Semantic Search               BM25 Search
        │                             │
        └──────────────┬──────────────┘
                       ▼
                 Hybrid Retrieval
                       │
                       ▼
              Similarity Graph
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
      Leiden                       UMAP
   Clustering                 Visualization
                       │
                       ▼
                Local LLM (Ollama)
                       │
                       ▼
                  Streamlit UI
```

---

## Technologies

### Literature management

- Zotero
- PyZotero

### PDF processing

- PyMuPDF
- OCRmyPDF (optional)
- Tesseract OCR

### Embeddings

- Sentence Transformers
- BAAI/bge-m3

### Search

- Cosine similarity
- BM25

### Graph analysis

- igraph
- Leiden algorithm

### Visualization

- UMAP
- Plotly
- Streamlit (planned)

### Local LLM

- Ollama
- Qwen 3

---

## Installation

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Required Software

- Zotero
- Tesseract OCR

- Ollama

---

## Models

### Embedding model

```
BAAI/bge-m3
```

### Local LLM

Currently tested with

```
qwen3:8b
```

---

## Repository Structure

```
LibraryAssistant/

├── app.py                  # Streamlit application
├── data.py                 # Load Zotero library
├── extraction.py           # PDF text extraction
├── embeddings.py           # Embedding generation
├── search.py               # Semantic + BM25 search
├── graph.py                # Similarity graph
├── visualization.py        # UMAP / Plotly
├── llm.py                  # Ollama interface
│
├── data/
│
├── requirements.txt
└── README.md
```

---

## Planned Features

### Retrieval

- [x] Semantic search
- [x] BM25
- [ ] Hybrid reranking
- [ ] Citation graph
- [ ] Metadata filters

### Visualization

- [x] UMAP
- [x] Leiden clustering
- [ ] Interactive Plotly graph
- [ ] Cluster inspection
- [ ] Citation visualization
- [ ] Filters
- [ ] Highlight search results
- [ ] Highlight connections on hover

### LLM

- [ ] Paper summarization
- [ ] Question answering
- [ ] Cross-paper comparison
- [ ] Automatic literature reviews

### User Interface

- [ ] Streamlit application
- [ ] PDF viewer
- [ ] Open paper from Zotero
- [ ] Search history
- [ ] Saved collections

---

## Long-Term Vision

The goal is to build a local-first AI research assistant capable of searching, organizing and reasoning over an entire scientific knowledge base, including:

- scientific papers
- books
- notes
- code
- reports
- presentations
- experimental data

while preserving privacy and allowing the use of local language models.

The tool shall be modular.


---

## Documentation

Each module has a readme file to explain the usage and background knowledge.