import io
import os
import json

import pandas as pd

from flask import send_file
from flask import redirect, url_for

import dash
import dash_table
import dash_auth
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from app import application
from apps import app1, app3, app2


def prepare_excel():
    # Create DF
    d = {"col1": [1, 2], "col2": [3, 4]}
    df = pd.DataFrame(data=d)

    # Convert DF
    buf = io.BytesIO()
    excel_writer = pd.ExcelWriter(buf, engine="xlsxwriter")
    df.to_excel(excel_writer, sheet_name="sheet1")
    excel_writer.save()
    excel_data = buf.getvalue()
    buf.seek(0)

    return buf


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#343A40",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H4("ПромПрогноз", className="display-8", style={'color': 'white'}),
        #html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode())),
        html.Hr(),
        html.P("Explore and Viz", className="display-8", style={'color': 'white', 'font-size': 12}),
        dbc.Nav(
            [
                dbc.NavLink("Explore tool", href="/", active="exact", id='nav-explore', style={'color': 'white'}),
                dbc.NavLink("EDA tool", href="/eda", active="exact", id='nav-eda', style={'color': 'white'}),
                dbc.NavLink("Reduce tool", href="/reduce", active="exact", id='nav-reduce', style={'color': 'white'}),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

VALID_USERNAME_PASSWORD_PAIRS = {
    'de': '2021'
}

# установка авторизации
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback([Output("page-content", "children"),
               Output("nav-explore", "active"),
               Output("nav-reduce", "active"),
               Output("nav-eda", "active"),],
              [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return app1.layout, True, "exact", "exact"
    if pathname == "/reduce":
        return app3.layout, "exact", True, "exact"
    if pathname == "/eda":
        return app2.layout, "exact", "exact", True
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    ), "exact"


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')