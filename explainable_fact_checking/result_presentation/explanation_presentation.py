import html

import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.express as px

# Function that iterate over avalable labels of an explanation and concateate in
# an html page the tables generated with the impact scores of each explanation
# scores using style_single_exp_list function. For each iteration also the claim
# is inserted as first element of the explanation with the impact equal to the intercept.
"""
Create a plot with plotly showing the predictions contained in exp.predict_proba and the class names in eco.class_names
then convert it into an html string
"""


def style_exp_to_html(exp):
    """
    html = pred_df.to_html()
    pred_df = pd.DataFrame(zip(exp.class_names, exp.predict_proba), columns=['class_names', 'predict_proba'])
    fig = px.bar(pred_df, x="predict_proba", y="class_names", orientation='h',
                 height=400,
                 )
    html += pyo.plot(fig, output_type='div')

    class_order = [exp.class_names.index('SUPPORTS'), exp.class_names.index('REFUTES'), exp.class_names.index('NOT ENOUGH INFO')]
    for i in class_order:
    label = exp.class_names[i]
    exp_as_list = exp.as_list(i)
    exp_as_list.insert(0, [f'[CLAIM] {exp.claim} [CLAIM]', exp.intercept[i]])
    html += style_single_exp_list(exp_as_list, caption=label)

    """
    pred_df = pd.DataFrame(zip(exp.class_names, exp.predict_proba), columns=['class_names', 'predict_proba'])
    html = """<head> <meta charset="UTF-8"></head>"""
    html += pred_df.to_html()
    fig = px.bar(pred_df, x="predict_proba", y="class_names", orientation='h',
                 height=400,
                 )
    html += pyo.plot(fig, output_type='div')

    class_order = [exp.class_names.index('SUPPORTS'), exp.class_names.index('REFUTES'),
                   exp.class_names.index('NOT ENOUGH INFO')]
    for i in class_order:
        label = exp.class_names[i]
        exp_as_list = exp.as_list(i)
        exp_as_list.insert(0, [f'[CLAIM] {exp.claim} [CLAIM]', exp.intercept[i]])
        html += style_single_exp_list(exp_as_list, caption=label)
    return html


def style_single_exp_list(exp_as_list, caption):
    df = pd.DataFrame(exp_as_list)
    df = df.apply(html.escape)
    df_styled = df.style.background_gradient(cmap='Reds', subset=[1])

    # Add a solid line to separate the header
    df_styled.set_table_styles([
        {'selector': 'th', 'props': [('border-bottom', '2px solid black'), ('border-top', '0px')]},
        {'selector': 'td', 'props': [('border', '0px'), ('vertical-align', 'middle'), ('padding', '0.5em 0.5em')]}
    ]).format(precision=3, thousands=".", decimal=",")

    # Add a caption
    df_styled.set_caption(caption)
    return df_styled.to_html()
