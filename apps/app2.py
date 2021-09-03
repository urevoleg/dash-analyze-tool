from app import app, server
import sys

BASEDIR = server.config['BASEDIR']
sys.path.append(BASEDIR)

import os
import datetime as dt
import json
from random import gauss

import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import power_transform

import matplotlib._color_data as mcd

import dash_table
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go
import plotly.express as px
from plotly.colors import n_colors
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


DATASET_FOLDER = os.path.join(os.path.dirname(__file__), "tmp")
files = {dataset.split('.')[0]: os.path.join(DATASET_FOLDER, dataset) for dataset in os.listdir(DATASET_FOLDER) if dataset.endswith('csv')}

colors = list(dict(mcd.TABLEAU_COLORS).values())


style_cell = {'color': 'black',
                    'border': '1px solid #D8D8D8',
                    'fontWeight': 'light',
                    'fontFamily': 'Times New Roman',
                    'fontSize': 14,
                    'textAlign': 'center',
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'minWidth': '10px', 'width': '10px', 'maxWidth': '10px',}
style_header = {'backgroundColor': '#D8D8D8',
                      'fontWeight': 'bold',
                      'fontFamily': 'Times New Roman',
                      'fontSize': 16,
                      'border': '1px solid black'}


def open_df(path, parse_date=False, sort=False, sort_by=None):
    """
    Начальные данные:
    - dt - столбец со временем
    - category_{1:N} - столбцы с категориями
    - остальные числовые столбцы

    :param path: путь до файла с данными
    :return: pd.DataFrame с предобработкой
    """
    if path.endswith('csv'):
        df = pd.read_csv(path)
    if path.endswith('xls') or path.endswith('xlsx'):
        df = pd.read_excel(path)

    if parse_date:
        df['dt'] = pd.to_datetime(df['dt'], format="%Y-%m-%d")
    else:
        df['dt'] = range(df.shape[0])

    if sort:
        df = df.sort_values(by=[sort_by])

    filtered_columns = df.select_dtypes([np.number]).columns.to_list() + df.filter(like='category_').columns.to_list()

    return df[filtered_columns]


layout = dbc.Container([
    dbc.Row([
            dbc.Col([html.Label('Выбор файла данных'),
                     dcc.Dropdown(
                         options=[{'label': n, 'value': n} for n in files.keys()],
                         value=list(files.keys())[0],
                         multi=False,
                         id='file_selector-3'
                     ),
                     html.Br()], width=12)]),
    dbc.Row([dbc.Col([html.Label('Выбор параметра'),
                     dcc.Dropdown(
                         multi=False,
                         id='feature_selector-3'
                     ),
                     html.Br()], width=3),
             dbc.Col([html.Label('Кол-во столбцов гистограммы'),
                      dcc.Slider(
                          id='slider_selector-3',
                          min=5,
                          max=51,
                          step=2,
                          value=11
                      )], width=3),
             dbc.Col([html.Label('Выбор типа графика:'),
                 dcc.RadioItems(
                     options=[
                         {'label': 'Гистограмма', 'value': 'hist'},
                         {'label': 'Квантиль-Квантиль', 'value': 'qqplot'},
                     ],
                     value='hist',
                     labelStyle={'display': 'block'},
                     id='hist_type_selector-3'
                 )], width=3),
            dbc.Col([html.Label('Преобразование:'),
                dcc.RadioItems(
                     options=[
                         {'label': 'No transform', 'value': 'no'},
                         {'label': 'Box-Cox', 'value': 'box'},
                         {'label': 'Yeo-Johnson', 'value': 'yeo'}
                     ],
                     value='no',
                     labelStyle={'display': 'block'},
                     id='transform_selector-3'
                 )
            ], width=3),
             ]),
        dbc.Row([
            # таблица describe
            dbc.Col(html.Div(id='live-update-table-3-0'), width=2),
            # гистограмма
            dbc.Col([dcc.Graph(id='live-update-3-0')], width=10),
            ]),
    dbc.Row([
        dbc.Col([dcc.Graph(id='live-update-3-1'), html.Br(), ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            html.Br(),
            html.Br(),
            html.Label('ось Х: '),
            dcc.Dropdown(
                multi=False,
                id='feature_selector_scatter_x-3'
            ),
            html.Br(),
            html.Label('ось Y: '),
            dcc.Dropdown(
                multi=False,
                id='feature_selector_scatter_y-3'
            ),
        ], width=4),
        dbc.Col([dcc.Graph(id='live-update-3-2'), html.Br(), ], width=8),
    ]),
    dcc.Store(id="store-app-3"),
    ])


@app.callback([Output("store-app-3", "data"),
               Output("feature_selector-3", "value"),
               Output("feature_selector-3", "options"),
               Output("feature_selector_scatter_x-3", "value"),
               Output("feature_selector_scatter_x-3", "options"),
               Output("feature_selector_scatter_y-3", "value"),
               Output("feature_selector_scatter_y-3", "options")],
              [Input('file_selector-3', 'value')])
def prefilter_data(n):
    print("prefilter_data", files[n])
    # заменить не на открытие, а чтение из store-app-3!
    data = open_df(files[n])
    data_filtered = data

    filter_nan = ~(data_filtered.isna().sum(axis=0) == data_filtered.shape[0])

    data_filtered_without_nan = data_filtered.loc[:, filter_nan]

    features_list = data_filtered_without_nan.filter(regex=r'^(?!category)|^(?!probe)').columns

    # features
    features = {'options': [dict(label=elem, value=elem) for elem in features_list],
               'value': features_list[0]}

    # features X
    features_x = {'options': [dict(label=elem, value=elem) for elem in features_list],
                'value': features_list[0]}

    # features Y
    features_y = {'options': [dict(label=elem, value=elem) for elem in features_list],
                'value': features_list[1]}

    return {
        'file': n,
        'data': data_filtered_without_nan[features_list].to_dict('list'),
    },\
        features['value'], \
        features['options'],\
        features_x['value'], \
        features_x['options'],\
        features_y['value'], \
        features_y['options']


# Multiple components can update everytime interval gets fired.
@app.callback([Output('live-update-3-0', 'figure'),
               Output("live-update-table-3-0", 'children')],
              [Input("store-app-3", "data"),
               Input('feature_selector-3', 'value'),
               Input("slider_selector-3", 'value'),
               Input("hist_type_selector-3", 'value'),
               Input("transform_selector-3", "value")])
def update_graph_live_0(data, param, slider_bins, hist_type, transform_type):
    print(param, slider_bins, hist_type, transform_type)
    # фильтруем датасет
    df = pd.DataFrame(data['data'])

    if transform_type == 'box':
        df.loc[:, f"{param}_transform"] = power_transform(df[param].values.reshape(-1, 1), method='box-cox')
    elif transform_type == 'yeo':
        df.loc[:, f"{param}_transform"] = power_transform(df[param].values.reshape(-1, 1), method='yeo-johnson')
    else:
        df.loc[:, f"{param}_transform"] = df[param]


    fig_param_0 = go.Figure()

    if hist_type == 'hist':
        fig_param_0.add_trace(go.Histogram(x=df[f"{param}"],
                                               nbinsx=slider_bins,
                                               marker=dict(color='#3690FF'),
                                               name=f"{param}"))
        Y_MAX = pd.cut(df[param], bins=slider_bins-1).value_counts().max()
        Y_MIN = 0
        title_header = "Гистограмма"
    else:
        # density
        # z-score param
        df.loc[:, f"{param}_zscore"] = zscore(df[f"{param}_transform"])
        fig_param_0.add_trace(go.Scatter(x=np.linspace(df[f"{param}_zscore"].min(), df[f"{param}_zscore"].max(), df.shape[0]),
                                             y=np.linspace(df[f"{param}_zscore"].min(), df[f"{param}_zscore"].max(), df.shape[0]),
                                             mode='lines',
                                             line=dict(color='red', width=0.5),
                                             name=f"Norm quantile"))

        # qqplot
        fig_param_0.add_trace(go.Scatter(x=np.linspace(df[f"{param}_zscore"].min(), df[f"{param}_zscore"].max(), df.shape[0]),
                                             y=df[f"{param}_zscore"].sort_values(),
                                             mode='markers',
                                             marker=dict(color='#3690FF', size=5),
                                             name=f"{param}_zscore"))
        Y_MAX = df[f"{param}_zscore"].max()
        Y_MIN = df[f"{param}_zscore"].min()
        title_header = "QQ-plot"

        # upper more 3
        fig_param_0.add_trace(go.Scatter(x=[2, 2, 3, 3], y=[Y_MIN, Y_MAX, Y_MAX, Y_MIN], fill="toself",
                                             mode="text", fillcolor='red', opacity=0.125, name='+3σ'))
        # upper between 2-3
        fig_param_0.add_trace(go.Scatter(x=[1, 1, 2, 2], y=[Y_MIN, Y_MAX, Y_MAX, Y_MIN], fill="toself",
                                             mode="text", fillcolor='yellow', opacity=0.125, name='+2σ'))
        # green -1 - 1
        fig_param_0.add_trace(go.Scatter(x=[-1, -1, 1, 1], y=[Y_MIN, Y_MAX, Y_MAX, Y_MIN], fill="toself",
                                             mode="text", fillcolor='green', opacity=0.125, name='1σ'))
        # lower between -2 - -3
        fig_param_0.add_trace(go.Scatter(x=[-1, -1, -2, -2], y=[Y_MIN, Y_MAX, Y_MAX, Y_MIN], fill="toself",
                                             mode="text", fillcolor='yellow', opacity=0.125, name='-2σ'))
        # lower low -3
        fig_param_0.add_trace(
                go.Scatter(x=[-2, -2, -3, -3], y=[Y_MIN, Y_MAX, Y_MAX, Y_MIN],
                           fill="toself",
                           mode="text", fillcolor='red', opacity=0.125, name='-3σ'))

    title_0 = f'{title_header} {param} (z-score)'

    fig_param_0.update_xaxes(tickfont=dict(family='Roboto', size=12, color='black'))
    fig_param_0.update_layout(title=dict(font=dict(family='Roboto', size=18, color='black'), text=title_0),
                                  template='plotly_white',
                                  margin=dict(l=0, r=0, t=40, b=10), yaxis_title=f'{param}',
                                  showlegend=True)

    decsribe_mapping = {
            'count': 'Размер выборки',
            'mean': 'Среднее',
            'std': 'Стд. откл.',
            'min': 'Минимум',
            '25%': '25% квартиль',
            '50%': 'Медиана',
            '75%': '75% квартиль',
            'max': 'Максимум',
        }

    descr = df[param].describe().to_frame().T.rename(columns=decsribe_mapping).T.reset_index()
    k_var = pd.DataFrame({'index': ['Коэффициент вариации'],
                              param: df[param].std() / df[param].mean()})

    descr_table = pd.concat([descr, k_var]).rename(columns={'index': 'Параметр'})
    descr_table[param] = np.round(descr_table[param], 3)

    return fig_param_0, [html.Label("Описательные статистики", style={'font-family': 'Roboto', 'font-size': 16, 'font-color': 'black'}),
                             html.Br(), dash_table.DataTable(
                        columns=[{"name": name_col, "id": name_col} for name_col in descr_table.columns],
                        data=descr_table.to_dict('records'),
                        style_as_list_view=False,
                        style_header=style_header,
                        style_cell=style_cell)]


# boxplot
@app.callback([Output('live-update-3-1', 'figure'),],
              [Input('feature_selector-3', 'value'),
               Input("store-app-3", "data")])
def update_graph_live_1(param, data):
    # фильтруем датасет
    df = pd.DataFrame(data['data'])

    fig_param_0 = go.Figure(data=[
                            go.Box(x=df[param],
                                   boxpoints='all',
                                   fillcolor='#98C6FF',
                                   line=dict(color='#3690FF'),
                                   marker=dict(opacity=0.5))
    ])

    title_0 = f'Диаграмма размаха: {param}'

    fig_param_0.update_xaxes(tickfont=dict(family='Roboto', size=12, color='black'))
    fig_param_0.update_layout(title=dict(font=dict(family='Roboto', size=18, color='black'), text=title_0), template='plotly_white',
                              height=240,
                              margin=dict(l=0, r=0, t=50, b=10), yaxis_title=f'{param}',
                              showlegend=False)

    return fig_param_0,


# scatter
@app.callback([Output('live-update-3-2', 'figure')],
              [Input('feature_selector_scatter_x-3', 'value'),
               Input('feature_selector_scatter_y-3', 'value'),
               Input("store-app-3", "data")])
def update_graph_live_2(xaxis, yaxis, data):
    # фильтруем датасет
    df = pd.DataFrame(data['data'])

    fig_param_0 = px.scatter(df, x=xaxis, y=yaxis)

    title_0 = f'Диаграмма рассеяния: {xaxis} vs {yaxis}'

    fig_param_0.update_xaxes(tickfont=dict(family='Roboto', size=12, color='black'))
    fig_param_0.update_layout(title=dict(font=dict(family='Roboto', size=18, color='black'), text=title_0), template='plotly_white',
                              margin=dict(l=0, r=0, t=100, b=10),
                              yaxis_title=f'{yaxis}',
                              xaxis_title=f'{xaxis}',
                              showlegend=True)

    return fig_param_0,