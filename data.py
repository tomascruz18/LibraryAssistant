"""Load paper metadata from a local Zotero library."""

from __future__ import annotations

from typing import Any

from pyzotero import zotero


def _creator_name(creator: dict[str, Any]) -> str:
    """Return a readable name from a Zotero creator record."""
    if creator.get("name"):
        return str(creator["name"])
    return " ".join(
        part
        for part in (creator.get("firstName", ""), creator.get("lastName", ""))
        if part
    )


def load_zotero_library(
    library_id: str = "0",
    library_type: str = "user",
    *,
    local: bool = True,
    limit: int | None = None,
    require_abstract: bool = False,
) -> list[dict[str, Any]]:
    """Return normalized paper records from Zotero.

    The local Zotero client must be running when ``local=True``. Attachments, notes,
    and annotations are omitted because they are not standalone papers.
    """
    client = zotero.Zotero(
        library_id=library_id,
        library_type=library_type,
        local=local,
    )
    items = client.everything(client.items())
    papers: list[dict[str, Any]] = []

    for item in items:
        item_data = item.get("data", {})
        if item_data.get("itemType") in {"attachment", "note", "annotation"}:
            continue

        abstract = str(item_data.get("abstractNote", "")).strip()
        if require_abstract and not abstract:
            continue

        creators = [
            name
            for creator in item_data.get("creators", [])
            if (name := _creator_name(creator))
        ]
        papers.append(
            {
                "id": item.get("key") or item_data.get("key"),
                "title": str(item_data.get("title", "")).strip(),
                "abstract": abstract,
                "date": str(item_data.get("date", "")).strip(),
                "authors": creators,
                "item_type": item_data.get("itemType", ""),
                "doi": str(item_data.get("DOI", "")).strip(),
                "url": str(item_data.get("url", "")).strip(),
            }
        )

        if limit is not None and len(papers) >= limit:
            break

    return papers
