from dash import Dash, html

app = Dash(__name__)
app.layout = html.Div("LibraryAssistant Dash app is running.")

if __name__ == "__main__":
    app.run(debug=True)