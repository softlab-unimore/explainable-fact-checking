
import sys, os, random
from sklearn.metrics import roc_curve
import numpy as np
import copy
import json

import explainable_fact_checking.experiment_definitions

print(sys.executable)
# insert '/home/bussotti/AB/code XFC' in the path
path = '/homes/bussotti/XFC2/code'
if path not in sys.path:
    sys.path.insert(0, path)
sys.path.insert(0, "/homes/bussotti/feverous_work/feverousdata/feverous/")
import explainable_fact_checking as xfc

save_path = xfc.experiment_definitions.C.PLOT_DIR

for extension in ['pdf', 'html', 'svg']:
    os.makedirs(os.path.join(save_path, extension), exist_ok=True)


import pandas as pd
import os
import plotly.express as px
# set color-schema with white background
import plotly.io as pio
from plotly.subplots import make_subplots

# check if in a colaboratory environment
if 'google.colab' in sys.modules:
    pio.renderers.default = 'colab'
import plotly.graph_objs as go

ratio = .9
pixel_width = 600
pixel_height = pixel_width / ratio

pio.templates["google"] = go.layout.Template(
    layout_colorway=['#4285F4', '#DB4437', '#F4B400', '#0F9D58',
                     '#185ABC', '#B31412', '#EA8600', '#137333',
                     '#d2e3fc', '#ceead6']
)
# new template with matplotlib color schema for plotly
pio.templates["matplotlib"] = go.layout.Template(
    layout_colorway=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                     '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                     '#bcbd22', '#17becf']
)

default_font_size = 16
layout = go.Layout(
    autosize=False,
    width=pixel_width,
    height=pixel_height,
    margin=go.layout.Margin(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
)
layout_dict = dict(autosize=False,
                   width=pixel_width,
                   height=pixel_height,
                   margin=go.layout.Margin(
                       l=55,
                       r=25,
                       b=55,
                       t=50,
                       pad=2
                   ),
                   font=dict(size=default_font_size),
                   )
h_legend_dict = dict(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.04,
    xanchor="right",
    x=1
))
pio.templates.default = "matplotlib"

pio.kaleido.scope.chromium_args = tuple([
    arg for arg in pio.kaleido.scope.chromium_args if arg != "--disable-dev-shm-usage"
])


def save_fig(fig, name, save_path=save_path):
    for extension in ['pdf', 'html', 'svg', 'png']:
        os.makedirs(os.path.join(save_path, extension), exist_ok=True)
    fig.write_html(os.path.join(save_path, 'html', name + '.html'))
    fig.write_image(os.path.join(save_path, 'svg', name + '.svg'))
    fig.write_image(os.path.join(save_path, 'pdf', name + '.pdf'))
    fig.write_image(os.path.join(save_path, 'png', name + '.png'))


title_off = True
columns_order_map = {
    'type': {'evidence': 0, 'claim_intercept': 1, 'only_claim': 2},
    'y': {'ground_truth': 0, 'predicted_label': 1},
    'model_id': {model_id: i for i, model_id in enumerate(['feverous_verdict_predictor', 'models_fromjf270623or'])},
    'explainer_name': {explainer_name: i for i, explainer_name in enumerate(['claim_only_pred', 'lime', 'shap', ])},
    'predicted_label': {class_: i for i, class_ in enumerate(
        explainable_fact_checking.experiment_definitions.CLASS_NAMES_V0)},
    'class': {class_: i for i, class_ in enumerate(explainable_fact_checking.experiment_definitions.CLASS_NAMES_V0)},

}


def sort_df(df, columns):
    to_sort = tuple(df[col].map(columns_order_map.get(col, sorted(df[col].unique()))) for col in columns)
    return df.iloc[np.lexsort(to_sort)]


def end_fig_func(fig):
    if title_off:
        fig.update_layout(title='')
    fig.for_each_annotation(lambda a: a.update(text=xfc.plot.style.replace_words(a.text)))
    fig.for_each_trace(lambda t: t.update(name=xfc.plot.style.replace_words(t.name),
                                          legendgroup=xfc.plot.style.replace_words(t.name),
                                          hovertemplate=xfc.plot.style.replace_words(t.hovertemplate),
                                          )
                       )
    for idx in range(len(fig.data)):
        fig.data[idx].x = [xfc.plot.style.replace_words(t) if isinstance(t, str) else t for t in fig.data[idx].x if t is not None]

    # increase space between yticks and y title

    # error 'NoneType' object is not iterable
    # fig.for_each_trace(lambda t: t.update(hovertemplate=xfc.plot.style.replace_words(t.hovertemplate)))

    # get x and y axis title
    # x_title = fig.layout.xaxis.title.text
    # y_title = fig.layout.yaxis.title.text
    # fig.for_each_xaxis(lambda x: x.update({'title': ''}))
    # fig.for_each_yaxis(lambda y: y.update({'title': ''}))
    # # add annotations to x and y axis use as reference the figure an bottom center of text for xaxis and left center for y axis
    # fig.add_annotation(
    #     showarrow=False,
    #     xanchor='center',
    #     yanchor='top',
    #     xref='paper',
    #     x=0.5,
    #     yref='paper',
    #     y=-0.05,
    #     text=x_title
    # )
    # fig.add_annotation(
    #     showarrow=False,
    #     xanchor='center',
    #     xref='paper',
    #     x=-0.05,
    #     yanchor='top',
    #     yref='paper',
    #     y=0.5,
    #     textangle=-90,
    #     text=y_title
    # )
    # # adjust margins
    # fig.update_layout(margin=dict(l=100, r=100, t=100, b=100))

    return fig