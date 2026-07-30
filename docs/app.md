## Description of app layout

Layout should be the same at the of the Figure App layout. 

There are three panels:

- the bottom panel is a chat
- the main panel is the visualization of the network
- the right panel is the info panel when we click on a paper

The bottom panel is a chat window to talk to the llm so we can ask questions about a paper, a group of papers, a cluster or the network. 
We can make this panel bigger or smaller my moving a grabbing the horizontal line. 
This panel always stays at the bottom

The main panel is where we visualize the network. 
At the center of the panel, at the bottom right above the chat panel there are two bottoms. 
One Force the other UMAP, so that we can switch between force directed visualization and UMAP visualization. 
We shall be able to zoom and pan the network.
Hovering above the nodes displays the title.
Clicking a node shows its information on the right panel and highlights the node.
On the top left corner there is a filter bottom. 
When clickling on it we can select the filters. 
For now we will only be able to filter by cluster using the cluster names. 
The cluster names should have a circle of the associated colour before the name.
On top of the main panel there should be a search bar. 
The search bard allows us to type in keywords to search.
When pressing enter a dropdown list should appear with the relevant papers.
there should be a dropdown list on the search bar to have the options to select hybrid, semantic or lexi. 
Defualt should be hybrid.
When clicking on a search result, the node should be highlighted and the info should appear on the right pannel.

The right panel contains the information of the last clicked node. 
At the beginning it shows empty fields.
We can make it bigger or smaller by moving the left vertical line left and right.
It should display all relevant ifnromation that is available.
there should be a buttom to open the pdf.

![App layout](img.png)

## App Roadmap

The application should use the existing SQLite catalog and persisted embeddings, graph,
clusters, labels, and projections. It should not rebuild the Zotero pipeline as part of
normal app use.

### Stage 1: Usable research-map MVP

Build the core three-panel application:

- main panel with a Force/UMAP switch, pan and zoom, title-only hover, and paper-node
  selection;
- right panel showing the selected paper's title, authors, date, abstract, cluster name,
  and Zotero key;
- search bar with Hybrid, Semantic, and BM25 modes; selecting a result also selects and
  highlights its node;
- cluster filter showing each saved cluster name with its associated colour;
- minimal chat panel, grounded in the selected paper, selected cluster, or retrieved
  papers from the database.

Two supporting pieces are required for this stage: expose force-layout coordinates for
the app instead of only writing them to HTML, and resolve a Zotero attachment key to a
local PDF path or an "Open in Zotero" action.

This stage is implemented in Dash. Start it from the project root with:

```powershell
python app.py
```

Then open `http://127.0.0.1:8050`. The application reads `data/library.sqlite3`;
generate or refresh that catalog separately with `scripts\build_all.py`.

### Stage 2: Interaction and usability polish

- draggable horizontal chat divider and vertical information-panel divider;
- distinct highlighting for search results, selected papers, and direct graph neighbours;
- cluster description and paper count in the filter and information panels;
- optional node sizing by graph degree or another centrality metric;
- session-only chat history and a visible processing/status area.

### Stage 3: Deeper research workflows

- multi-select papers and compare them through the chat;
- LLM answers with paper citations and clickable source cards;
- manual cluster-name editing, preserving manual labels over automatic proposals;
- metadata filters for date, author, document type, and metadata provenance;
- embedded PDF viewer, saved collections, annotations, and search history.

### Later extensions

- incremental Zotero synchronization from the interface;
- citation-network ingestion and visualization;
- graph editing and user-defined links;
- cross-paper comparison and automatic literature-review workflows.
