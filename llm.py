"""Small Ollama interface for answers grounded in retrieved papers."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

import ollama


DEFAULT_MODEL = "qwen3:8b"
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_CONTEXT_WINDOW_TOKENS = 8_000
METADATA_PIPELINE_VERSION = 1
# Scientific PDFs contain equations and symbols that may tokenize more densely than
# ordinary prose. Two characters per token is intentionally conservative.
CONSERVATIVE_CHARACTERS_PER_TOKEN = 2
METADATA_OUTPUT_TOKENS = 700
DEFAULT_METADATA_MAX_ATTEMPTS = 3
SUMMARY_OUTPUT_TOKENS = 400
PROMPT_RESERVE_TOKENS = 600
METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "authors": {"type": "array", "items": {"type": "string"}},
        "date": {"type": ["string", "null"]},
        "abstract": {"type": ["string", "null"]},
        "document_type": {
            "type": "string",
            "enum": ["research_paper", "presentation_or_lecture", "other"],
        },
    },
    "required": ["authors", "date", "abstract", "document_type"],
    "additionalProperties": False,
}
AUTHOR_FOOTNOTE_MARKERS = r"*+#$&†‡§¶‖"


def _clean_author_name(author: str) -> str:
    """Remove trailing author-affiliation markers, not letters within a name."""
    cleaned = author.strip().rstrip(",;")
    return re.sub(rf"\s*[{re.escape(AUTHOR_FOOTNOTE_MARKERS)}]+$", "", cleaned).strip()


class MetadataExtractionError(RuntimeError):
    """Raised when every metadata extraction attempt fails validation."""

    def __init__(
        self,
        attempts: int,
        last_error: Exception,
        raw_response: str | None,
    ) -> None:
        super().__init__(
            f"Metadata extraction failed after {attempts} attempts: {last_error}"
        )
        self.attempts = attempts
        self.last_error = last_error
        self.raw_response = raw_response


def parse_metadata_response(content: str) -> dict[str, Any]:
    """Parse and validate the constrained JSON response from Ollama."""
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)

    metadata = json.loads(text)
    expected_fields = {"authors", "date", "abstract", "document_type"}
    if not isinstance(metadata, dict) or set(metadata) != expected_fields:
        raise ValueError(f"Expected exactly these fields: {sorted(expected_fields)}")
    if not isinstance(metadata["authors"], list) or not all(
        isinstance(author, str) for author in metadata["authors"]
    ):
        raise ValueError("'authors' must be a list of strings.")
    if metadata["date"] is not None and not isinstance(metadata["date"], str):
        raise ValueError("'date' must be a string or null.")
    if metadata["abstract"] is not None and not isinstance(metadata["abstract"], str):
        raise ValueError("'abstract' must be a string or null.")
    if metadata["document_type"] not in {
        "research_paper",
        "presentation_or_lecture",
        "other",
    }:
        raise ValueError("'document_type' has an unsupported value.")

    return {
        "authors": [
            cleaned
            for author in metadata["authors"]
            if (cleaned := _clean_author_name(author))
        ],
        "date": metadata["date"].strip() if metadata["date"] else None,
        "abstract": metadata["abstract"].strip() if metadata["abstract"] else None,
        "document_type": metadata["document_type"],
    }


def request_metadata_from_text(
    pdf_text: str,
    *,
    model: str = DEFAULT_MODEL,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
) -> str:
    """Request structured metadata from Ollama and return its unparsed text response."""
    prompt = f"""Extract bibliographic metadata from the beginning of this scientific paper.

Rules:
- Use only the supplied text. Never invent authors, a date, or an abstract.
- Classify the document as `research_paper`, `presentation_or_lecture`, or `other`.
  A slide deck, lecture, course handout, or title slide is `presentation_or_lecture`.
  For that type and for `other`, always return abstract: null.
- Preserve author order and names as shown in the paper, but remove trailing
  author-affiliation or footnote markers such as *, +, #, $, &, dagger, and double
  dagger. Do not include institutions, job titles, addresses, or email addresses.
- Use the publication date exactly as shown. If it is not present, return null.
- Extract the paper's author-provided abstract faithfully. An "Abstract" heading is
  common but not required: also accept a concise stand-alone opening paragraph placed
  after the title/authors/affiliations and before the first numbered or named section
  (for example, "Introduction") when it states the work's purpose, approach, or
  findings. Do not mistake affiliations, copyright, keywords, or the first body
  paragraph of a section for an abstract. Never treat a slide title, running header,
  footer, agenda, course title, or repeated presentation text as an abstract. Return
  null only when no abstract-like opening paragraph is present.
- Respond with JSON only, matching the requested schema.

PDF text:
---
{pdf_text}
---
"""
    client = ollama.Client(timeout=timeout_seconds)
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=METADATA_SCHEMA,
        think=False,
        options={
            "temperature": 0,
            "num_ctx": context_window_tokens,
            "num_predict": METADATA_OUTPUT_TOKENS,
        },
    )
    return str(response["message"]["content"])


def extract_metadata_from_text(
    pdf_text: str,
    *,
    model: str = DEFAULT_MODEL,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
    max_attempts: int = DEFAULT_METADATA_MAX_ATTEMPTS,
) -> dict[str, Any]:
    """Extract authors, date, and an explicitly present abstract from PDF text.

    The result is deliberately limited to the three fields used by the MVP. Missing
    evidence must be represented by an empty author list or a null field. It does not
    summarize a paper: that is a separate operation.
    """
    metadata, _attempts, _raw_response = extract_metadata_with_retries(
        pdf_text,
        model=model,
        timeout_seconds=timeout_seconds,
        context_window_tokens=context_window_tokens,
        max_attempts=max_attempts,
    )
    return metadata


def extract_metadata_with_retries(
    pdf_text: str,
    *,
    model: str = DEFAULT_MODEL,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
    max_attempts: int = DEFAULT_METADATA_MAX_ATTEMPTS,
) -> tuple[dict[str, Any], int, str]:
    """Request and validate metadata, retrying transient model failures only."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    raw_response: str | None = None
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw_response = request_metadata_from_text(
                pdf_text,
                model=model,
                timeout_seconds=timeout_seconds,
                context_window_tokens=context_window_tokens,
            )
            return parse_metadata_response(raw_response), attempt, raw_response
        except Exception as error:
            last_error = error

    assert last_error is not None
    raise MetadataExtractionError(max_attempts, last_error, raw_response) from last_error


def _summary_chunk_characters(context_window_tokens: int) -> int:
    """Return a conservative input size that leaves prompt/output space in context."""
    available_tokens = (
        context_window_tokens - PROMPT_RESERVE_TOKENS - SUMMARY_OUTPUT_TOKENS
    )
    if available_tokens <= 0:
        raise ValueError("context_window_tokens is too small for document summarization.")
    return available_tokens * CONSERVATIVE_CHARACTERS_PER_TOKEN


def _split_text(text: str, maximum_characters: int) -> list[str]:
    """Split text near paragraph or sentence boundaries without exceeding the budget."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    if len(normalized) <= maximum_characters:
        return [normalized]

    pieces = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if len(piece) > maximum_characters:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                piece[index : index + maximum_characters]
                for index in range(0, len(piece), maximum_characters)
            )
            continue
        candidate = f"{current} {piece}".strip()
        if current and len(candidate) > maximum_characters:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _pack_texts(texts: Sequence[str], maximum_characters: int) -> list[str]:
    """Combine short texts into the largest context-safe groups possible."""
    groups: list[str] = []
    current = ""
    for text in texts:
        for piece in _split_text(text, maximum_characters):
            candidate = f"{current}\n\n{piece}".strip()
            if current and len(candidate) > maximum_characters:
                groups.append(current)
                current = piece
            else:
                current = candidate
    if current:
        groups.append(current)
    return groups


def _summarize_chunk(
    text: str,
    *,
    model: str,
    client: ollama.Client,
    context_window_tokens: int,
) -> str:
    prompt = f"""Write a concise, factual abstract-style summary of the scientific text below.

Use only the supplied text. Cover the problem, approach, and key findings when present.
Do not claim a result that the text does not support. Limit the summary to 180 words.

Text:
---
{text}
---
"""
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        think=False,
        options={
            "temperature": 0,
            "num_ctx": context_window_tokens,
            "num_predict": SUMMARY_OUTPUT_TOKENS,
        },
    )
    summary = str(response["message"]["content"]).strip()
    if not summary:
        raise ValueError("The model returned an empty summary.")
    return summary


def summarize_document_text(
    document_text: str,
    *,
    model: str = DEFAULT_MODEL,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
) -> str:
    """Summarize a document of any length using recursive context-safe reduction.

    Each source chunk is summarized independently. If the resulting summaries still do
    not fit safely in one request, they are summarized again until one final summary
    remains. This keeps every inference within the configured context window.
    """
    chunk_characters = _summary_chunk_characters(context_window_tokens)
    source_chunks = _split_text(document_text, chunk_characters)
    if not source_chunks:
        raise ValueError("Cannot summarize empty document text.")

    client = ollama.Client(timeout=timeout_seconds)
    if len(source_chunks) == 1:
        return _summarize_chunk(
            source_chunks[0],
            model=model,
            client=client,
            context_window_tokens=context_window_tokens,
        )

    summaries = [
        _summarize_chunk(
            chunk,
            model=model,
            client=client,
            context_window_tokens=context_window_tokens,
        )
        for chunk in source_chunks
    ]
    while len(summaries) > 1:
        summary_groups = _pack_texts(summaries, chunk_characters)
        summaries = [
            _summarize_chunk(
                chunk,
                model=model,
                client=client,
                context_window_tokens=context_window_tokens,
            )
            for chunk in summary_groups
        ]

    return summaries[0]


def answer_question(
    question: str,
    papers: Sequence[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
) -> str:
    """Ask Ollama to answer using the supplied paper excerpts."""
    context_parts = []
    for index, paper in enumerate(papers, start=1):
        context_parts.append(
            f"[{index}] {paper.get('title', 'Untitled')}\n"
            f"{paper.get('abstract', '')}"
        )
    context = "\n\n".join(context_parts)
    prompt = (
        "Answer the question using only the sources below. Cite sources with bracketed "
        "numbers such as [1]. If the sources are insufficient, say so.\n\n"
        f"Question: {question}\n\nSources:\n{context}"
    )
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return str(response["message"]["content"])
