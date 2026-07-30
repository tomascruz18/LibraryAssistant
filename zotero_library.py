from pyzotero import zotero

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
            "year": data.get("date", "")
        })

    if len(papers) == 10:
        break

for paper in papers:

    print("="*80)
    print(paper["title"])
    print()
    print(paper["abstract"][:500])