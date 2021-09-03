import os

import flask

import dash
import dash_auth
import dash_bootstrap_components as dbc

from dotenv import load_dotenv
load_dotenv()

# bootstrap theme
# https://bootswatch.com/lux/
external_stylesheets = [dbc.themes.BOOTSTRAP]

server = flask.Flask(__name__) # define flask app.server
# add config file to app
server.config.from_pyfile("flask.cfg")

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}],
                server=server)
app.title = "Explore analytical tool"
application = app.server
app.config.suppress_callback_exceptions = True

