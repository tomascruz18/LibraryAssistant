"""Local Dash MVP for exploring the persisted LibraryAssistant catalog."""

from __future__ import annotations

import os
from typing import Any

from dash import Dash, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from flask import abort, send_file
import plotly.graph_objects as go

from app_data import LibraryAppData
from llm import DEFAULT_MODEL, answer_question


try:
    LIBRARY = LibraryAppData.load()
    STARTUP_ERROR: str | None = None
except Exception as error:
    LIBRARY = None
    STARTUP_ERROR = str(error)


app = Dash(__name__, title="LibraryAssistant", update_title="Working…")
server = app.server


def _cluster_options() -> list[dict[str, Any]]:
    if LIBRARY is None:
        return []
    return [
        {
            "label": html.Span(
                [
                    html.Span(
                        className="cluster-dot",
                        style={"backgroundColor": cluster.color},
                    ),
                    html.Span(
                        f"{cluster.label} ({cluster.paper_count})",
                        className="cluster-option-label",
                    ),
                ],
                className="cluster-option",
            ),
            "value": cluster.cluster_id,
        }
        for cluster in LIBRARY.clusters.values()
    ]


def _empty_info_panel() -> html.Div:
    return html.Div(
        [
            html.Div("Paper details", className="panel-kicker"),
            html.H2("Nothing selected", className="empty-title"),
            html.P(
                "Select a node or a search result to inspect its metadata.",
                className="muted-copy",
            ),
        ],
        className="info-empty",
    )


def _startup_error_layout(message: str) -> html.Div:
    return html.Div(
        [
            html.Div("LibraryAssistant", className="brand-mark"),
            html.Div(
                [
                    html.Div("Database unavailable", className="panel-kicker"),
                    html.H1("Build the local research index first"),
                    html.P(message, className="muted-copy"),
                    html.Code(
                        "python scripts\\build_all.py --papers 1000",
                        className="setup-command",
                    ),
                ],
                className="setup-card",
            ),
        ],
        className="startup-shell",
    )


def _main_layout() -> html.Div:
    initial_figure = LIBRARY.figure("umap") if LIBRARY else go.Figure()
    return html.Div(
        [
            dcc.Store(id="selected-paper"),
            dcc.Store(id="chat-history", data=[]),
            html.Div(
                [
                    html.Main(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                "LibraryAssistant",
                                                className="brand-mark",
                                            ),
                                            html.Div(
                                                f"{len(LIBRARY.papers)} papers",
                                                className="library-count",
                                            ),
                                        ],
                                        className="brand-block",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Dropdown(
                                                id="search-mode",
                                                options=[
                                                    {
                                                        "label": "Hybrid",
                                                        "value": "hybrid",
                                                    },
                                                    {
                                                        "label": "Semantic",
                                                        "value": "semantic",
                                                    },
                                                    {
                                                        "label": "Lexical",
                                                        "value": "bm25",
                                                    },
                                                ],
                                                value="hybrid",
                                                clearable=False,
                                                searchable=False,
                                                className="search-mode",
                                            ),
                                            dcc.Input(
                                                id="search-input",
                                                type="search",
                                                placeholder="Search the library…",
                                                debounce=False,
                                                className="search-input",
                                            ),
                                            html.Button(
                                                "Search",
                                                id="search-submit",
                                                n_clicks=0,
                                                className="search-button",
                                            ),
                                            html.Div(
                                                id="search-status",
                                                className="search-status",
                                            ),
                                            dcc.Dropdown(
                                                id="search-results",
                                                options=[],
                                                placeholder="Select a result",
                                                className="search-results",
                                                style={"display": "none"},
                                            ),
                                        ],
                                        className="search-block",
                                    ),
                                    html.Div(className="header-spacer"),
                                ],
                                className="graph-toolbar",
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        [
                                            html.Span("⌁", className="filter-icon"),
                                            "Clusters",
                                        ],
                                        id="filter-button",
                                        n_clicks=0,
                                        className="filter-button",
                                    ),
                                    dbc.Collapse(
                                        html.Div(
                                            [
                                                html.Div(
                                                    "Filter by cluster",
                                                    className="filter-heading",
                                                ),
                                                dcc.Dropdown(
                                                    id="cluster-filter",
                                                    options=_cluster_options(),
                                                    multi=True,
                                                    placeholder="All clusters",
                                                    className="cluster-filter",
                                                ),
                                            ],
                                            className="filter-card",
                                        ),
                                        id="filter-collapse",
                                        is_open=False,
                                    ),
                                ],
                                className="filter-control",
                            ),
                            dcc.Graph(
                                id="library-graph",
                                figure=initial_figure,
                                config={
                                    "displaylogo": False,
                                    "scrollZoom": True,
                                    "responsive": True,
                                    "modeBarButtonsToRemove": [
                                        "select2d",
                                        "lasso2d",
                                    ],
                                },
                                className="library-graph",
                            ),
                            dbc.RadioItems(
                                id="layout-mode",
                                options=[
                                    {"label": "Force", "value": "force"},
                                    {"label": "UMAP", "value": "umap"},
                                ],
                                value="umap",
                                className="layout-switch",
                                inputClassName="layout-radio-input",
                                labelClassName="layout-option",
                                labelCheckedClassName="layout-option-active",
                            ),
                        ],
                        className="graph-panel",
                    ),
                    html.Aside(
                        _empty_info_panel(),
                        id="info-panel",
                        className="info-panel",
                    ),
                ],
                className="top-workspace",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.Div("Research chat", className="chat-title"),
                            html.Div(
                                "Grounded in the selected paper, filtered clusters, "
                                "or retrieved library results.",
                                className="chat-scope",
                            ),
                        ],
                        className="chat-header",
                    ),
                    dcc.Loading(
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span(
                                            "Assistant", className="message-author"
                                        ),
                                        html.P(
                                            "Select a paper or ask a question about "
                                            "your library.",
                                            className="message-text",
                                        ),
                                    ],
                                    className="chat-message assistant-message",
                                )
                            ],
                            id="chat-messages",
                            className="chat-messages",
                        ),
                        type="circle",
                    ),
                    html.Div(
                        [
                            dcc.Input(
                                id="chat-input",
                                type="text",
                                placeholder="Ask about this paper, cluster, or library…",
                                className="chat-input",
                            ),
                            html.Button(
                                "Send",
                                id="chat-send",
                                n_clicks=0,
                                className="chat-send",
                            ),
                        ],
                        className="chat-composer",
                    ),
                ],
                className="chat-panel",
            ),
        ],
        className="app-shell",
    )


app.layout = (
    _startup_error_layout(STARTUP_ERROR)
    if STARTUP_ERROR
    else _main_layout()
)


@server.route("/pdf/<zotero_key>")
def serve_pdf(zotero_key: str) -> Any:
    if LIBRARY is None or zotero_key not in LIBRARY.papers_by_key:
        abort(404)
    path = LIBRARY.pdf_path(zotero_key)
    if path is None:
        abort(404)
    return send_file(path, as_attachment=False, download_name=path.name)


@app.callback(
    Output("filter-collapse", "is_open"),
    Input("filter-button", "n_clicks"),
    State("filter-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_filter(_clicks: int, is_open: bool) -> bool:
    return not is_open


@app.callback(
    Output("library-graph", "figure"),
    Input("layout-mode", "value"),
    Input("cluster-filter", "value"),
    Input("selected-paper", "data"),
)
def update_graph(
    layout_mode: str,
    cluster_ids: list[int] | None,
    selected_key: str | None,
) -> go.Figure:
    if LIBRARY is None:
        return go.Figure()
    return LIBRARY.figure(layout_mode, cluster_ids, selected_key)


@app.callback(
    Output("search-results", "options"),
    Output("search-results", "value"),
    Output("search-results", "style"),
    Output("search-status", "children"),
    Input("search-submit", "n_clicks"),
    Input("search-input", "n_submit"),
    State("search-input", "value"),
    State("search-mode", "value"),
    prevent_initial_call=True,
)
def run_search(
    _clicks: int,
    _submits: int,
    query: str | None,
    mode: str,
) -> tuple[list[dict[str, str]], None, dict[str, str], str]:
    if LIBRARY is None or not query or not query.strip():
        raise PreventUpdate
    try:
        results = LIBRARY.search(query, mode, limit=12)
    except Exception as error:
        return [], None, {"display": "none"}, f"Search failed: {error}"
    options = [
        {
            "label": f"{result['title']}  ·  {result['score']:.3f}",
            "value": result["id"],
        }
        for result in results
    ]
    style = {"display": "block"} if options else {"display": "none"}
    return options, None, style, f"{len(options)} results"


@app.callback(
    Output("selected-paper", "data"),
    Input("library-graph", "clickData"),
    Input("search-results", "value"),
    prevent_initial_call=True,
)
def select_paper(
    click_data: dict[str, Any] | None,
    search_key: str | None,
) -> str:
    if ctx.triggered_id == "search-results" and search_key:
        return search_key
    if ctx.triggered_id == "library-graph" and click_data:
        custom_data = click_data["points"][0].get("customdata")
        if custom_data:
            return str(custom_data[0])
    raise PreventUpdate


@app.callback(
    Output("info-panel", "children"),
    Input("selected-paper", "data"),
)
def update_info_panel(zotero_key: str | None) -> html.Div:
    if LIBRARY is None or not zotero_key:
        return _empty_info_panel()
    paper = LIBRARY.papers_by_key.get(zotero_key)
    if paper is None:
        return _empty_info_panel()
    cluster = LIBRARY.cluster_for_key(zotero_key)
    pdf_available = LIBRARY.pdf_path(zotero_key) is not None
    authors = ", ".join(paper.get("authors", [])) or "Unknown"
    return html.Div(
        [
            html.Div("Selected paper", className="panel-kicker"),
            html.H2(paper.get("title") or "Untitled", className="paper-title"),
            html.Div(
                [
                    html.Span(
                        className="cluster-dot cluster-dot-large",
                        style={
                            "backgroundColor": (
                                cluster.color if cluster else "#c7cbd1"
                            )
                        },
                    ),
                    html.Span(
                        cluster.label if cluster else "Unclustered",
                        className="paper-cluster",
                    ),
                ],
                className="paper-cluster-row",
            ),
            html.Div(
                [
                    html.Div("Authors", className="metadata-label"),
                    html.Div(authors, className="metadata-value"),
                    html.Div("Date", className="metadata-label"),
                    html.Div(
                        paper.get("date") or "Unknown",
                        className="metadata-value",
                    ),
                    html.Div("Type", className="metadata-label"),
                    html.Div(
                        paper.get("item_type") or "Unknown",
                        className="metadata-value",
                    ),
                    html.Div("Zotero key", className="metadata-label"),
                    html.Div(zotero_key, className="metadata-value metadata-code"),
                    html.Div("DOI", className="metadata-label"),
                    html.Div(
                        paper.get("doi") or "Not available",
                        className="metadata-value",
                    ),
                ],
                className="metadata-grid",
            ),
            html.Div("Abstract / summary", className="section-label"),
            html.P(
                paper.get("abstract") or "No abstract is available.",
                className="paper-abstract",
            ),
            html.Div(
                [
                    dbc.Button(
                        "Open PDF",
                        href=f"/pdf/{zotero_key}" if pdf_available else None,
                        target="_blank",
                        disabled=not pdf_available,
                        className="info-action primary-action",
                    ),
                    dbc.Button(
                        "Open in Zotero",
                        href=f"zotero://select/library/items/{zotero_key}",
                        className="info-action secondary-action",
                    ),
                ],
                className="info-actions",
            ),
        ],
        className="paper-info",
    )


def _render_chat(history: list[dict[str, str]]) -> list[html.Div]:
    if not history:
        return [
            html.Div(
                [
                    html.Span("Assistant", className="message-author"),
                    html.P(
                        "Select a paper or ask a question about your library.",
                        className="message-text",
                    ),
                ],
                className="chat-message assistant-message",
            )
        ]
    return [
        html.Div(
            [
                html.Span(
                    "You" if message["role"] == "user" else "Assistant",
                    className="message-author",
                ),
                dcc.Markdown(message["content"], className="message-text"),
            ],
            className=(
                "chat-message user-message"
                if message["role"] == "user"
                else "chat-message assistant-message"
            ),
        )
        for message in history
    ]


@app.callback(
    Output("chat-history", "data"),
    Output("chat-messages", "children"),
    Output("chat-input", "value"),
    Input("chat-send", "n_clicks"),
    Input("chat-input", "n_submit"),
    State("chat-input", "value"),
    State("chat-history", "data"),
    State("selected-paper", "data"),
    State("cluster-filter", "value"),
    prevent_initial_call=True,
)
def send_chat_message(
    _clicks: int,
    _submits: int,
    question: str | None,
    history: list[dict[str, str]] | None,
    selected_key: str | None,
    cluster_ids: list[int] | None,
) -> tuple[list[dict[str, str]], list[html.Div], str]:
    if LIBRARY is None or not question or not question.strip():
        raise PreventUpdate
    question = question.strip()
    history = list(history or [])
    if selected_key and selected_key in LIBRARY.papers_by_key:
        sources = [LIBRARY.papers_by_key[selected_key]]
    elif cluster_ids:
        sources = LIBRARY.papers_in_clusters(cluster_ids, limit=8)
    else:
        sources = LIBRARY.search(question, "hybrid", limit=8)

    try:
        answer = answer_question(question, sources, model=DEFAULT_MODEL)
        source_list = "\n".join(
            f"[{index}] {paper.get('title', 'Untitled')}"
            for index, paper in enumerate(sources, start=1)
        )
        if source_list:
            answer = f"{answer}\n\n**Sources used**\n{source_list}"
    except Exception as error:
        answer = f"I could not complete that request: {error}"

    history.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )
    return history, _render_chat(history), ""


if __name__ == "__main__":
    app.run(debug=os.environ.get("DASH_DEBUG") == "1")
