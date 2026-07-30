# LibraryAssistant
A local AI-powered literature assistant for exploring, searching and reasoning over a personal scientific paper library.

The project combines semantic search, keyword search, graph analysis and local large language models to create an interactive research environment.

> [!IMPORTANT]
> **Project status:** this repository is an early MVP. Reusable building blocks follow
> the flat module structure below, while the original exploratory code is preserved in
> `scripts/`. A more elaborate possible design is documented separately in
> [Future Architecture and Roadmap](docs/FUTURE_ARCHITECTURE_AND_ROADMAP.md).

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

## Run the MVP

Start Zotero with its local API enabled, then launch the application:

```bash
streamlit run app.py
```

Use **Load Zotero and build index** in the sidebar. The MVP indexes paper titles and
abstracts in memory, so the index is rebuilt when the application process restarts.

### Test LLM metadata extraction from one PDF

With Ollama running and `qwen3:8b` installed, run:

```bash
python scripts/llm_metadata_from_pdf.py "path/to/paper.pdf" --mode auto
```

The script reads the beginning of the PDF, then prints JSON with `authors`, `date`, and
`abstract`. It first extracts a real abstract only when one is explicitly present. If
there is none, it generates an abstract-style summary from the complete document.
Long documents are summarized in context-safe chunks and recursively condensed using the
8k context configuration. Pass `--output data/metadata.json` to save the result. For
scanned PDFs, run OCR first; this small experiment currently uses only the PDF's
embedded text.

To compare the two LLM operations on the same PDF, use `--mode extract` for strict
front-matter extraction and `--mode summarize` to force a full-document summary.

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

Maximum context size is 40960. After some experiments I got this:

| Context |  Memory | GPU usage |
| ------: | ------: | --------: |
|      4k |  5.6 GB |  100% GPU |
|      8k |  6.2 GB |  100% GPU |
|     12k |  7.2 GB |   87% GPU |
|     16k |  7.8 GB |   80% GPU |
|     30k | 10.0 GB |   62% GPU |


---

## Repository Structure

```
LibraryAssistant/

├── app.py                  # Streamlit application
├── data.py                 # Load Zotero library
├── storage.py              # Persist metadata in SQLite
├── extraction.py           # PDF text extraction
├── embeddings.py           # Embedding generation
├── search.py               # Semantic + BM25 search
├── graph.py                # Similarity graph
├── visualization.py        # UMAP / Plotly
├── llm.py                  # Ollama interface
│
├── data/                   # Generated local MVP data
├── scripts/                # Original standalone experiments
├── docs/
│
├── requirements.txt
└── README.md
```

---

## Planned Features

### Retrieval

- [x] Semantic search MVP
- [x] BM25 search MVP
- [x] Hybrid score fusion MVP
- [ ] Persistent indexes
- [ ] Reranking
- [ ] Citation graph
- [ ] Metadata filters

### Visualization

- [x] UMAP prototype
- [x] Leiden clustering prototype
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

- [x] Minimal Streamlit search application
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

Longer design notes and future plans live in [`docs/`](docs/). The root modules contain
short docstrings for their current APIs.

- [LLM metadata extraction and summarization](docs/llm.md)
- [Paper data model](docs/data.md)
- [SQLite storage and database inspection](docs/storage.md)
