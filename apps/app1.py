from app import app, server
import sys

BASEDIR = server.config['BASEDIR']
sys.path.append(BASEDIR)

import os
import datetime as dt
import json

import base64
import io

import pandas as pd
import numpy as np
from scipy.stats import zscore

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def prepare_df(df, parse_date=False, sort=False, sort_by=None):
    """
    Начальные данные:
    - dt - столбец со временем
    - category_{1:N} - столбцы с категориями
    - остальные числовые столбцы

    :param path: путь до файла с данными
    :return: pd.DataFrame с предобработкой
    """
    if parse_date:
        df['dt'] = pd.to_datetime(df['dt'])
    else:
        df['dt'] = range(df.shape[0])

    if sort:
        df = df.sort_values(by=sort_by)

    columns = ['dt'] + df.select_dtypes(include=np.number).columns.tolist()

    return df[columns]


layout = dbc.Container([
    dbc.Row([
            dbc.Col([dcc.Upload(
                        id='upload-data-1',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        # Allow multiple files to be uploaded
                        multiple=False
                    )], width=6),
            dbc.Col([html.Div(id="upload-filename-1"),
                     html.Br()], width=6)
                ]),
    dbc.Row([dbc.Col([html.Label('Выбор параметра'),
                     dcc.Dropdown(
                         multi=False,
                         id='feature_selector'
                     ),
                     html.Br()], width=12),]),
    dbc.Row([
        dbc.Row([
            dbc.Col([dcc.Graph(id='live-update-0'), html.Br(), ], width=12),
            dbc.Col([dcc.Graph(id='live-update-2'), html.Br(), ], width=12),
            dbc.Col([dcc.Graph(id='live-update-3'), html.Br(), ], width=12),
            ]),
        dcc.Store(id="store-app-999")
        ])
    ])


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        return html.Div([
            'There was an error processing this file.'
        ])
    print(df.head())
    return df, filename


@app.callback([Output("store-app-999", "data"),
               Output("feature_selector", "value"),
               Output("feature_selector", "options"),
               Output("upload-filename-1", "children")],
              [Input('upload-data-1', 'contents')],
              [State('upload-data-1', 'filename'),
              State('upload-data-1', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    print('Upload!')
    print(list_of_names)
    if list_of_contents is not None:
        df, filename = parse_contents(list_of_contents, list_of_names)
        df = prepare_df(df, parse_date=True, sort=True, sort_by='dt')
        features = {'options': [dict(label=elem, value=elem) for elem in df.select_dtypes(include=np.number).columns],
                    'value': df.select_dtypes(include=np.number).columns[0]}
        return {'file': filename,
                'data': df.to_dict('list')}, \
        features['value'], \
        features['options'],\
        html.Div([html.Br(),
                     html.H5(f'Загружен файл: {filename}')])


# Multiple components can update everytime interval gets fired.
@app.callback([Output('live-update-0', 'figure'),
               Output('live-update-2', 'figure'),
               Output('live-update-3', 'figure')],
              [Input('feature_selector', 'value'),
               Input("store-app-999", "data")])
def update_graph_live(param, data):
    # фильтруем датасет
    df = pd.DataFrame(data['data'])
    # исходные данные
    fig_param_0 = go.Figure(data=go.Scatter(x=df['dt'], y=df[param], name='Измерения', mode='lines+markers',
                                            line=dict(width=0.25, color='grey'),
                                            marker=dict(size=3, color='red')))
    fig_param_0.add_trace(
        go.Scatter(x=df['dt'], y=[df[param].mean()] * df.shape[0], name='Среднее<br>значение', mode='lines',
                   line=dict(width=0.25, color='blue')))

    fig_param_0.update_xaxes(tickfont=dict(family='Roboto', size=12, color='black'))
    fig_param_0.update_layout(title=f'Параметр: {param}, {data["file"]} (исходные семплы)', template='plotly_white',
                              margin=dict(l=0, r=0, t=50, b=10), yaxis_title=f'{param}',
                              showlegend=True)

    # разности первого порядка - скользящий размах в терминах SPC
    fig_param_2 = go.Figure(data=go.Scatter(x=[*range(df.shape[0])], y=df[param].diff(), name='Скользящий размах', mode='lines+markers',
                                            line=dict(width=0.25, color='grey'),
                                            marker=dict(size=3, color='red')))
    fig_param_2.add_trace(
        go.Scatter(x=[*range(df.shape[0])], y=[df[param].diff().mean()] * df.shape[0], name='Среднее<br>значение', mode='lines',
                   line=dict(width=0.25, color='blue')))

    fig_param_2.update_xaxes(tickfont=dict(family='Roboto', size=12, color='black'))
    fig_param_2.update_layout(title=f'Параметр: {param} (Скользящий размах)', template='plotly_white',
                              margin=dict(l=0, r=0, t=50, b=10), yaxis_title=f'{param}',
                              showlegend=True)

    # z-score param
    df.loc[:, f"{param}_zscore"] = zscore(df[param])
    fig_param_3 = go.Figure()

    # upper more 3
    fig_param_3.add_trace(go.Scatter(x=[0, df.shape[0]-1, df.shape[0]-1, 0, 0], y=[2, 2, df[f"{param}_zscore"].max(), df[f"{param}_zscore"].max(), 2], fill="toself",
                                     mode="text", fillcolor='red', opacity=0.125, name='+3σ'))
    # upper between 2-3
    fig_param_3.add_trace(go.Scatter(x=[0, df.shape[0] - 1, df.shape[0] - 1, 0, 0], y=[1, 1, 2, 2, 1], fill="toself",
                                     mode="text", fillcolor='yellow', opacity=0.125, name='+2σ'))
    # green -1 - 1
    fig_param_3.add_trace(go.Scatter(x=[0, df.shape[0] - 1, df.shape[0] - 1, 0, 0], y=[-1, -1, 1, 1, -1], fill="toself",
                                     mode="text", fillcolor='green', opacity=0.125, name='1σ'))
    # lower between -2 - -3
    fig_param_3.add_trace(go.Scatter(x=[0, df.shape[0] - 1, df.shape[0] - 1, 0, 0], y=[-1, -1, -2, -2, -1], fill="toself",
                                     mode="text", fillcolor='yellow', opacity=0.125, name='-2σ'))
    # lower low -3
    fig_param_3.add_trace(go.Scatter(x=[0, df.shape[0] - 1, df.shape[0] - 1, 0, 0], y=[-2, -2, df[f"{param}_zscore"].min(), df[f"{param}_zscore"].min(), -2], fill="toself",
                                     mode="text", fillcolor='red', opacity=0.125, name='-3σ'))

    fig_param_3.add_trace(go.Scatter(x=[*range(df.shape[0])], y=df[f"{param}_zscore"], name='Z-score', mode='lines+markers',
                        line=dict(width=0.25, color='grey'),
                        marker=dict(size=3, color='black')))

    title_3 = f'Параметр: {param} (Z-score)<br>\
    Стд.окл.: {df[param].std():.3f}<br>\
    Среднее.: {df[param].mean():.3f}<br>\
    Вариация: {df[param].std() / df[param].mean():.3f}'

    fig_param_3.update_xaxes(tickfont=dict(family='Roboto', size=12, color='black'))
    fig_param_3.update_layout(title=title_3, template='plotly_white',
                              margin=dict(l=0, r=0, t=150, b=10), yaxis_title=f'{param}',
                              showlegend=True)


    return fig_param_0, fig_param_3, fig_param_2