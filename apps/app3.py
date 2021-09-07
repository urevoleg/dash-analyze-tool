from app import app, server
import sys

BASEDIR = server.config['BASEDIR']
sys.path.append(BASEDIR)

DATASETDIR = server.config['DATASETDIR']

IS_MEMCACHED = server.config['IS_MEMCACHED']

# comment
import os
import json

from pymemcache.client import base

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from umap import UMAP
from sklearn.mixture import GaussianMixture

import matplotlib._color_data as mcd

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go


colors = list(dict(mcd.TABLEAU_COLORS).values())

umap_metrics = ['euclidean', 'manhattan', 'mahalanobis', 'seuclidean', 'cosine', 'correlation', 'jaccard', 'yule']

files = {dataset.split('.')[0]: os.path.join(DATASETDIR, dataset) for dataset in os.listdir(DATASETDIR) if dataset.endswith('csv')}


class DictCashe(type({})):
    def add(self, key, value, **kwargs):
        self.update({key: value})


def make_cach_client(host='localhost', port=11211):
    if IS_MEMCACHED:
        return base.Client((host, port))
    else:
        if globals().get('c') is None:
            global c
            c = DictCashe()
            return c
        else:
            return globals().get('c')


# масштабирование без масштабирования
class DummyScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array(X)


# dummy dim reduction
class DimReduction(BaseEstimator, TransformerMixin):
    """
    in: числовые признаки для понижения размерности
    out: массив признаков указанной размерности (пока реализовано только 2)
    """
    def __init__(self, n_components=1, random_state=2021):
        self.n_components = n_components
        self.X_transform_ = None
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()  # creating a copy to avoid changes to original dataset
        size = X_.shape[1]
        self.X_transform_ = np.column_stack([X_[:, :size // 2].mean(axis=1),
                         X_[:, size // 2:].mean(axis=1)])
        return self.X_transform_


methods = dict(zip(['PCA', 'MDS', 't-SNE', 'Dummy', 'UMAP'],
                   [PCA(n_components=2, random_state=2021),
                    MDS(n_components=2, random_state=2021, n_jobs=-1),
                    TSNE(n_components=2, random_state=2021, n_jobs=-1),
                    DimReduction(n_components=2),
                    UMAP(n_components=2, random_state=2021, n_jobs=-1)]))


scalers = dict(zip(['StandardScaler',
                    'MinMax',
                    'MaxAbsScaler',
                    'RobustScaler',
                    'DummyScaler'],
                   [StandardScaler(),
                    MinMaxScaler(),
                    MaxAbsScaler(),
                    RobustScaler(quantile_range=(5, 95)),
                    DummyScaler()]))


def prepare_df(path, parse_date=False, sort=False, sort_by=None):
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
        df['dt'] = pd.to_datetime(df['dt'])
    else:
        df['dt'] = range(df.shape[0])

    if sort:
        df = df.sort_values(by=[sort_by])
        df = df.drop(sort_by, axis=1)

    return df


def reduct_dimension(df, method, scaler):
    pipeline = Pipeline([('Preprocessing (StandardScaler)', scalers[scaler]),
                         ('Dim reduction method', methods[method])], verbose=False)
    df_dim = pipeline.fit_transform(df)
    res = pd.DataFrame(df_dim, columns=[f'{method}_1', f'{method}_2'])

    return res


layout = dbc.Container([
    dbc.Row([
            dbc.Col([html.Label('Выбор файла данных'),
                     dcc.Dropdown(
                         options=[{'label': n, 'value': n} for n in files.keys()],
                         value=list(files.keys())[0],
                         multi=False,
                         id='file_selector-2'
                     ),
                     html.Br()], width=12)]),
    dbc.Row([
        dbc.Col([html.Label('Выбор метода понижения размерности:'),
                 dcc.Dropdown(
                     options=[{"label": k, "value": k} for k in methods.keys()],
                     value='PCA',
                     multi=False,
                     id='method_selector-2'
                 ),
                 html.Br()], width=3),
        dbc.Col([html.Label('Выбор метода масштабирования:'),
                 dcc.Dropdown(
                     options=[{"label": k, "value": k} for k in scalers.keys()],
                     value='StandardScaler',
                     multi=False,
                     id='scaler_selector-2'
                 ),
                 html.Br()], width=3),
        dbc.Col([html.Label('Кол-во кластеров:'),
                 dcc.Dropdown(
                     options=[{"label": k, "value": k} for k in range(1, 10)],
                     value=1,
                     multi=False,
                     id='slider_selector-2'
                 ),
                 html.Br()], width=3),
        dbc.Col([html.Label('Вращение к Х:'),
                 dcc.RadioItems(
                     options=[
                         {'label': 'Rotate', 'value': 1},
                         {'label': 'No rotate', 'value': 0}
                     ],
                     value=0,
                     id='rotate-selector',
                     labelStyle={'display': 'block'}
                 ),
                 html.Br()], width=3)
    ]),
    html.Div(
            [
                dbc.Button(
                    "Setup of reduction method",
                    id="collapse-button",
                    className="mb-3",
                    color="primary",
                    n_clicks=0,
                ),
                dbc.Collapse(
                    dbc.Row([
                        dbc.Col([html.Label('Perplexity:'),
                                 dcc.Slider(
                                     min=5,
                                     max=50,
                                     step=5,
                                     marks={k: str(k) for k in range(5, 50, 5)},
                                     value=30
                                 )], width=12),
                        dbc.Col([html.Label('N-neighbours:'),
                                 dcc.Slider(
                                     min=5,
                                     max=150,
                                     step=10,
                                     marks={k: f'{k}' for k in range(5, 150, 10)},
                                     value=15,
                                     id='umap-neighbours'
                                 )], width=4),
                        dbc.Col([html.Label('Min dist:'),
                                 dcc.Slider(
                                     min=0.1,
                                     max=1.0,
                                     step=0.1,
                                     marks={k: f'{k:.2f}' for k in np.linspace(0.1, 1.0, 10)},
                                     value=0.2,
                                     id='umap-mindist'
                                 )], width=4),
                        dbc.Col([html.Label('Metric:'),
                                 dcc.Dropdown(
                                     options=[{"label": k, "value": k} for k in umap_metrics],
                                     value=umap_metrics[0],
                                     multi=False,
                                     id='umap-metric'
                                 )], width=4)
                    ]),
                    id="collapse",
                    is_open=False,
                ),
            ]
        ),
        html.Br(),
    dbc.Row([
            dbc.Col(dcc.Graph(id='live-update-0-2'), width=12),
            dcc.Store(id="store-app-2"),
        ])
    ])


@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback([Output("store-app-2", "data")],
              [Input('file_selector-2', 'value')])
def choose_file(n):
    print(files[n])
    data = prepare_df(files[n])

    return {
        'file': n,
        'data': data.to_dict('list'),
        'time_line': 'dt'
    },


# Multiple components can update everytime interval gets fired.
@app.callback([Output('live-update-0-2', 'figure')],
              [Input('method_selector-2', 'value'),
               Input('scaler_selector-2', 'value'),
               Input("store-app-2", "data"),
               Input("slider_selector-2", "value"),
               Input("rotate-selector", "value"),
               Input("umap-neighbours", "value"),
               Input("umap-mindist", "value"),
               Input("umap-metric", "value")])
def update_graph_live(method, scaler, data, n_clusters, is_rotate, umap_neighbours, umap_mindist, umap_metric):

    hash_req = '-'.join([str(elem) for elem in [hash(json.dumps(data['data'])), method, scaler, n_clusters, is_rotate,
                                                umap_neighbours, umap_mindist, umap_metric]])

    # фильтруем датасет
    df = pd.DataFrame(data['data']).drop(data['time_line'], axis=1)
    print(df.head())

    if method == 'UMAP':
        methods['UMAP'] = UMAP(n_components=2, n_neighbors=umap_neighbours, min_dist=umap_mindist,
                               metric=umap_metric, random_state=88)

    """res = reduct_dimension(df.sample(frac=1, replace=False, random_state=114), method, scaler)

    gm = GaussianMixture(n_components=n_clusters, random_state=88)
    gm.fit(res.values)
    res['gr'] = gm.predict(res.values)"""

    # caching request
    # put to cach
    c = make_cach_client()

    if c.get(hash_req) is None:
        print('Is new data!')
        res = reduct_dimension(df.sample(frac=1, replace=False, random_state=114), method, scaler)

        gm = GaussianMixture(n_components=n_clusters, random_state=88)
        gm.fit(res.values)
        res['gr'] = gm.predict(res.values)

        c.add(hash_req, json.dumps(res.to_dict('records')), expire=60*10)
    else:
        print('Is old data!')
        res = pd.DataFrame(json.loads(c.get(hash_req)))


    # исходные данные
    data_s = []
    shapes = []
    for idx, gr in enumerate(sorted(res['gr'].unique())):
        tmp = res.query("gr==@gr")

        if is_rotate:
            # calculate rotation angle
            theta = np.arctan((tmp[f'{method}_2'].max() - tmp[f'{method}_2'].min()) / (tmp[f'{method}_1'].max() - tmp[f'{method}_1'].min()))

            # calculate rotation matrix
            rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                      [np.sin(theta), np.cos(theta)]])

            # rotate
            tmp_rotate = tmp.values[:, :2].dot(rotate_matrix)
            tmp.iloc[:, 0], tmp.iloc[:, 1] = tmp_rotate[:, 0], tmp_rotate[:, 1]

        data_s += [go.Scatter(x=tmp[f'{method}_1'], y=tmp[f'{method}_2'], name=f"Group: {gr}", mode='markers',
                                                marker=dict(size=5, color=colors[idx], opacity=0.45),
                              text=tmp.reset_index()['index'].map(lambda x: str(f'Sample index: {x}')))]

        # add cluster centers

        #print(idx, gm.means_[idx], gm.covariances_[idx][:2, :2])
        #add centers
        data_s += [go.Scatter(x=[tmp.iloc[:, 0].mean()], y=[tmp.iloc[:, 1].mean()], name=f"Center group: {idx}", mode='markers',
                              marker=dict(size=10, color=colors[idx]),
                              opacity=0.75)]

        # add clusters contour with shapes
        # color contour
        shapes.append(
                dict(
                    type="circle",
                    xref="x",
                    yref="y",
                    x0=tmp.iloc[:, 0].mean() - 1.5 * tmp.iloc[:, 0].std(),
                    y0=tmp.iloc[:, 1].mean() - 1.5 * tmp.iloc[:, 1].std(),
                    x1=tmp.iloc[:, 0].mean() + 1.5 * tmp.iloc[:, 0].std(),
                    y1=tmp.iloc[:, 1].mean() + 1.5 * tmp.iloc[:, 1].std(),
                    opacity=0.75,
                    fillcolor=None,
                    line_color=colors[idx],
                ))

    fig_param_0 = go.Figure(data=data_s)

    fig_param_0.update_xaxes(tickfont=dict(family='Roboto', size=12, color='black'))
    fig_param_0.update_layout(title=f'{data["file"]}', template='plotly_white',
                              margin=dict(l=0, r=0, t=50, b=10), yaxis_title=f'{method}_2',
                              xaxis_title=f'{method}_1',
                              showlegend=True)

    return fig_param_0,