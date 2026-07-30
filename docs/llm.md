# LLM Metadata Extraction and Document Summarization

This module fills missing paper metadata from a PDF using a local Ollama model. It is
used by `data.py` and can also be tested directly with
`scripts/llm_metadata_from_pdf.py`.

The current MVP extracts three fields:

```json
{
  "authors": ["First Author", "Second Author"],
  "date": "2006",
  "abstract": "Paper abstract or generated abstract-style summary"
}
```

## What happens during library loading

`data.load_zotero_library()` takes Zotero metadata as the primary source. It calls the
LLM only when `authors`, `date`, or `abstract` is missing.

```text
Zotero item
    |
    +-- All three fields present --> return Zotero metadata unchanged
    |
    +-- One or more fields absent
            |
            +-- Find the first PDF attachment
            +-- Extract embedded PDF text
            +-- Strictly extract authors, date, and an existing abstract from front matter
            |
            +-- No abstract found?
                    |
                    +-- Summarize the full document in context-safe chunks
```

Before abstract extraction, the same LLM response classifies the document as a
`research_paper`, `presentation_or_lecture`, or `other`. Presentations and lectures are
excluded from the returned paper list by default, so title-slide text cannot enter the
paper catalog as an abstract. Pass `include_unsupported_documents=True` only when
inspecting such records deliberately.

The integration is read-only. It does not write generated data back to Zotero. Returned
paper records show where each value came from:

```python
paper["metadata_source"]
# {
#   "authors": "zotero" or "llm_extracted",
#   "date": "zotero" or "llm_extracted",
#   "abstract": "zotero", "llm_extracted", or "llm_summary"
# }
```

If an attachment cannot be found, has no embedded text, or Ollama fails, the paper is
still returned and the reason is stored in `paper["metadata_error"]`.

## Strict metadata extraction

`llm.extract_metadata_from_text()` receives only the opening PDF text. Its job is
extraction, not writing. It returns:

- authors, in the order shown on the paper;
- the publication or conference date as written;
- a faithful abstract, but only when the PDF contains one.

An abstract heading is not required. The prompt also recognizes an unlabeled,
stand-alone opening paragraph between the title/author block and the first paper section,
such as `Introduction`. This is common in conference PDFs.

For example, the Fedorov RD-170 conference paper has no `Abstract` label, but its bold
paragraph immediately before `I. Introduction` is correctly kept as
`"llm_extracted"`, rather than being rewritten as a summary.

### Author affiliation markers

Papers often attach symbols to author names to link them to affiliations or footnotes:

```text
V. Fedorov*, V. Chvanov+, F. Chelkis#
```

The model is instructed to omit these symbols. `llm._clean_author_name()` also removes
trailing `*`, `+`, `#`, `$`, `&`, dagger, double-dagger, and similar markers after the
model responds. This deterministic cleanup protects the result when PDF text extraction
or the model retains a marker.

## Full-document summarization

`llm.summarize_document_text()` is used only when strict extraction returns
`"abstract": null`.

The project uses an 8k-token context window to keep inference on the GPU. A full paper
may be much larger, so the module does not put the whole PDF into one request.

Instead, it uses recursive map/reduce summarization:

1. Split the document into conservative context-safe chunks.
2. Ask the model for a factual, abstract-style summary of each chunk.
3. Group the partial summaries into context-safe batches.
4. Summarize those batches again until one final summary remains.

The exact Ollama/Qwen tokenizer is not exposed by the installed Python client. The
module therefore uses a cautious estimate of two characters per token, reserves space
for the prompt and answer, and sends at most about 14,000 characters per source chunk
with an 8k context. Scientific text, equations, and symbols can tokenize more densely
than normal prose, so the conservative limit is intentional.

The generated text is labelled `"llm_summary"`; it must never be mistaken for an
author-provided abstract.

## Unsupported documents

This MVP is designed for research papers, not lecture slides or presentation decks. A
document classified as `presentation_or_lecture` or `other` is not summarized during
normal loading and is excluded before it reaches the SQLite paper catalog. The one-PDF
test script can still use `--mode summarize` when explicitly requested for experiments.

## Important code locations

| File | Relevant parts |
|---|---|
| `llm.py` | `extract_metadata_from_text()` for strict front-matter extraction; `_clean_author_name()` for affiliation-marker removal; `summarize_document_text()` for recursive full-document summarization. |
| `data.py` | `_enrich_missing_metadata()` ties Zotero attachment retrieval, strict extraction, and fallback summarization together. |
| `scripts/llm_metadata_from_pdf.py` | Command-line experiment for testing one PDF without Zotero. |

## Testing one PDF

Run commands from the project root with Zotero and Ollama available. The script does
not modify the PDF or Zotero.

### 1. Strict extraction only

Use this to check whether the paper has an author-provided abstract, including an
unlabeled one.

```powershell
python scripts\llm_metadata_from_pdf.py "C:\path\to\paper.pdf" --mode extract
```

Expected outcomes:

```json
{
  "abstract": "Exact or faithful author-provided abstract",
  "abstract_source": "llm_extracted"
}
```

or, when no abstract is found:

```json
{
  "abstract": null,
  "abstract_source": null
}
```

### 2. Force a generated summary

Use the same PDF to test the full-document, context-safe summarization path. This does
not attempt to extract an existing abstract first.

```powershell
python scripts\llm_metadata_from_pdf.py "C:\path\to\paper.pdf" --mode summarize
```

Expected output contains:

```json
{
  "abstract": "Generated abstract-style summary",
  "abstract_source": "llm_summary"
}
```

### 3. Normal application behavior

This is the mode used conceptually by `data.py`: extract first, summarize only when an
abstract is absent.

```powershell
python scripts\llm_metadata_from_pdf.py "C:\path\to\paper.pdf" --mode auto
```

Save any result for comparison with `--output`:

```powershell
python scripts\llm_metadata_from_pdf.py "C:\path\to\paper.pdf" --mode extract --output data\metadata-extracted.json
python scripts\llm_metadata_from_pdf.py "C:\path\to\paper.pdf" --mode summarize --output data\metadata-summary.json
```

When debugging a malformed or truncated JSON response, save Ollama's unparsed response
before the script attempts to parse it:

```powershell
python scripts\llm_metadata_from_pdf.py "C:\path\to\paper.pdf" --mode extract --raw-output data\raw-llm-response.txt
```

The raw text is saved even if JSON parsing subsequently fails. `--raw-output` is for
metadata extraction (`extract` or `auto`) and is not used with forced summarization.

## Retry behavior

Metadata extraction makes up to three attempts when Ollama fails to respond or returns
invalid JSON/schema data. A successful result includes `metadata_llm_attempts`, so a
value of `1` means no retry was needed. PDF download, native-text extraction, and OCR
are not retried.

## Testing through Zotero loading

To test the integration rather than a standalone PDF, ask for a small number of records:

```powershell
python -c "from data import load_zotero_library; papers = load_zotero_library(limit=10, llm_timeout_seconds=120); [print(p['id'], p['metadata_source'], p.get('metadata_error')) for p in papers]"
```

Inspect a particular item:

```powershell
python -c "from data import load_zotero_library; papers = load_zotero_library(limit=10); paper = next(p for p in papers if p['id'] == 'ITEMKEY'); print(paper['abstract']); print(paper['metadata_source']); print(paper.get('metadata_error'))"
```

## Examples

```powershell
python scripts\llm_metadata_from_pdf.py "C:\Users\tr-mo\Zotero\storage\24Y6CEWC\2.pdf" --mode extract --output data\extracted_metadata.json
```

```json
{
  "authors": [
    "Prof. Dr.-Ing. O. J. Haidn",
    "Dipl.-Ing. Andrej Sternin",
    "Daniel Marinez M. Sc."
  ],
  "date": "Introduction 1",
  "abstract": null,
  "document_type": "presentation_or_lecture",
  "abstract_source": null
}
```


```powershell
python scripts\llm_metadata_from_pdf.py "C:\Users\tr-mo\Zotero\storage\42UIS42K\2-missions.pdf" --mode extract --output data\extracted_metadata.json
```

```json
{
  "authors": [
    "Prof. Dr.-Ing. O.J. Haidn"
  ],
  "date": "Winter Term 2020/2021",
  "abstract": null,
  "document_type": "presentation_or_lecture",
  "abstract_source": null
}
```

```powershell
python scripts\llm_metadata_from_pdf.py "C:\Users\tr-mo\Zotero\storage\M7YT5VSL\TEC-FRA-DOC-2025-01515-2-huracan_mcc5_design.pdf" --mode extract --output data\extracted_metadata.json  
```

```json
{
  "authors": [
    "Matteo Crachi",
    "Tomas Cruz",
    "Dimitrios Vogiatzief"
  ],
  "date": "2025/12/05",
  "abstract": "This document details the main design modiﬁcations of MCC5. The document goes into detail on the optimization process of the cooling channels. The optimization process is the direct follow up of the learnings from TCA4 campaign [RD1] and the root cause analysis of the MCC4 cracks [RD2]. Brief overview of minor modiﬁcations is also given.",
  "document_type": "research_paper",
  "text_extraction_method": "native",
  "abstract_source": "llm_extracted"
}
```

```powershell
python scripts\llm_metadata_from_pdf.py "C:\Users\tr-mo\Zotero\storage\A2ISCMT9\Fedorov et al. - 2006 - The Chamber Cooling System of RD-170 Engine Family Design, Parameters, and Hardware Investigation D.pdf" --mode extract --output data\extracted_metadata.json ```
```

```json
{
  "authors": [
    "V. Fedorov",
    "V. Chvanov",
    "F. Chelkis",
    "А. Polyansky",
    "N. Ivanov",
    "I. Lozino-Lozinskaya",
    "А. Buryak"
  ],
  "date": "2006-4363",
  "abstract": "Up to now many aspects of a multiuse booster stage for the next generation rockets still remain the subject of research and discussions. One of the issues is selection of propellant, where amongst its selection criteria is possibility of insuring reliable chamber cooling during multiple flights. This article describes the main structural elements and parameters of the chamber cooling passages for RD-170 family of LOX/kerosene engines. Also, herein are evaluation results of chamber hardware after multiple hot fire tests.",
  "document_type": "research_paper",
  "text_extraction_method": "native",
  "abstract_source": "llm_extracted"
}
```



## Current limits

- PDFs are extracted with native text first. When that produces fewer than 100
  characters, OCRmyPDF is run automatically with deskewing and page rotation. OCR
  requires the local OCRmyPDF and Tesseract installation.
- Only the first PDF attachment for an item is considered.
- Full-document fallback summarization is skipped above an estimated 24,000 tokens
  (about 40 scientific-paper pages). The estimate uses two extracted characters per
  token, so it errs on the side of skipping long documents such as books. Strict
  front-matter metadata extraction still runs for these files. The unified pipeline can
  override this with `--max-paper-tokens`.
- The LLM result is useful metadata, not verified bibliographic truth; inspect samples
  before using it to update Zotero in a future workflow.
- The implementation intentionally does not deduplicate Zotero records or copy metadata
  between duplicate items.
