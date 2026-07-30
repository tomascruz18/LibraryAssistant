"""Small Ollama interface for answers grounded in retrieved papers."""

from __future__ import annotations

from typing import Any, Sequence

import ollama


DEFAULT_MODEL = "qwen3:8b"


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
