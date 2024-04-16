#todo delete this file

import inspect
import json
import os

import pandas as pd
from ipywidgets import HTML
from lime.explanation import id_generator
import lime
from rich.jupyter import display
from sklearn.utils import check_random_state


def as_html_perturb_evidence(exp, labels=None, predict_proba=True, show_predicted_value=True, **kwargs):
    def jsonize(x):
        return json.dumps(x, ensure_ascii=False)

    labels = exp.available_labels() if labels is None and exp.mode == "classification" else labels

    module_path = inspect.getfile(lime.explanation)
    this_dir = os.path.dirname(module_path)
    bundle = open(f"{this_dir}/bundle.js", encoding="utf8").read()
    random_id = id_generator(size=15, random_state=check_random_state(exp.random_state))

    predict_proba_js = f"""
    var pp_div = top_div.append('div').classed('lime predict_proba', true);
    var pp_svg = pp_div.append('svg').style('width', '100%%');
    var pp = new lime.PredictProba(pp_svg, {jsonize([str(x) for x in exp.class_names])}, {jsonize(list(exp.predict_proba.astype(float)))});
    """ if exp.mode == "classification" and predict_proba else ''

    predict_value_js = f'''
    var pp_div = top_div.append('div').classed('lime predicted_value', true);
    var pp_svg = pp_div.append('svg').style('width', '100%%');
    var pp = new lime.PredictedValue(pp_svg, {jsonize(float(exp.predicted_value))}, {jsonize(float(exp.min_value))}, {jsonize(float(exp.max_value))});
    ''' if exp.mode == "regression" and show_predicted_value else ''

    exp_js = f'''var exp_div;
    var exp = new lime.Explanation({jsonize([str(x) for x in exp.class_names])});
    '''

    if exp.mode == "classification":
        for label in labels:
            exp_js += f'''
            exp_div = top_div.append('div').classed('lime explanation', true);
            exp.show({jsonize(exp.as_list(label))}, {label}, exp_div);
            '''
    else:
        exp_js += f'''
        exp_div = top_div.append('div').classed('lime explanation', true);
        exp.show({jsonize(exp.as_list())}, {exp.dummy_label}, exp_div);
        '''

    raw_js = '''var raw_div = top_div.append('div');'''
    html_data = exp.local_exp[labels[0]] if exp.mode == "classification" else exp.local_exp[exp.dummy_label]
    raw_js += exp.domain_mapper.visualize_instance_html(html_data,
                                                        labels[0] if exp.mode == "classification" else exp.dummy_label,
                                                        'raw_div', 'exp', **kwargs)

    out = f'''
    <html>
    <meta http-equiv="content-type" content="text/html; charset=UTF8">
    <head><script>{bundle} </script></head><body>
    <div class="lime top_div" id="top_div{random_id}"></div>
    <script>
    var top_div = d3.select('#top_div{random_id}').classed('lime top_div', true);
    {predict_proba_js}
    {predict_value_js}
    {exp_js}
    {raw_js}
    </script>
    <h2>{exp.claim}</h2>
    </body></html>
    '''

    return out


def as_html_perturb_evidence_old(exp,
                                 labels=None,
                                 predict_proba=True,
                                 show_predicted_value=True,
                                 **kwargs):
    """Returns the explanation as an html page.

    Args:
        labels: desired labels to show explanations for (as barcharts).
            If you ask for a label for which an explanation wasn't
            computed, will throw an exception. If None, will show
            explanations for all available labels. (only used for classification)
        predict_proba: if true, add  barchart with prediction probabilities
            for the top classes. (only used for classification)
        show_predicted_value: if true, add  barchart with expected value
            (only used for regression)
        kwargs: keyword arguments, passed to domain_mapper

    Returns:
        code for an html page, including javascript includes.
    """

    def jsonize(x):
        return json.dumps(x, ensure_ascii=False)

    if labels is None and exp.mode == "classification":
        labels = exp.available_labels()

    this_dir, _ = os.path.split(__file__)
    bundle = open(os.path.join(this_dir, 'bundle.js'),
                  encoding="utf8").read()

    out = u'''<html>
    <meta http-equiv="content-type" content="text/html; charset=UTF8">
    <head><script>%s </script></head><body>''' % bundle
    random_id = id_generator(size=15, random_state=check_random_state(exp.random_state))
    out += u'''
    <div class="lime top_div" id="top_div%s"></div>
    ''' % random_id

    predict_proba_js = ''
    if exp.mode == "classification" and predict_proba:
        predict_proba_js = f"""
        var pp_div = top_div.append('div')
                            .classed('lime predict_proba', true);
        var pp_svg = pp_div.append('svg').style('width', '100%%');
        var pp = new lime.PredictProba(pp_svg, {jsonize([str(x) for x in exp.class_names])}, {jsonize(list(exp.predict_proba.astype(float)))});
        """

    predict_value_js = ''
    if exp.mode == "regression" and show_predicted_value:
        # reference exp.predicted_value
        # (svg, predicted_value, min_value, max_value)
        predict_value_js = u'''
                var pp_div = top_div.append('div')
                                    .classed('lime predicted_value', true);
                var pp_svg = pp_div.append('svg').style('width', '100%%');
                var pp = new lime.PredictedValue(pp_svg, %s, %s, %s);
                ''' % (jsonize(float(exp.predicted_value)),
                       jsonize(float(exp.min_value)),
                       jsonize(float(exp.max_value)))

    exp_js = '''var exp_div;
        var exp = new lime.Explanation(%s);
    ''' % (jsonize([str(x) for x in exp.class_names]))

    if exp.mode == "classification":
        for label in labels:
            exp = jsonize(exp.as_list(label))
            exp_js += u'''
            exp_div = top_div.append('div').classed('lime explanation', true);
            exp.show(%s, %d, exp_div);
            ''' % (exp, label)
    else:
        exp = jsonize(exp.as_list())
        exp_js += u'''
        exp_div = top_div.append('div').classed('lime explanation', true);
        exp.show(%s, %s, exp_div);
        ''' % (exp, exp.dummy_label)

    raw_js = '''var raw_div = top_div.append('div');'''

    if exp.mode == "classification":
        html_data = exp.local_exp[labels[0]]
    else:
        html_data = exp.local_exp[exp.dummy_label]

    raw_js += exp.domain_mapper.visualize_instance_html(
        html_data,
        labels[0] if exp.mode == "classification" else exp.dummy_label,
        'raw_div',
        'exp',
        **kwargs)
    out += u'''
    <script>
    var top_div = d3.select('#top_div%s').classed('lime top_div', true);
    %s
    %s
    %s
    %s
    </script>
    ''' % (random_id, predict_proba_js, predict_value_js, exp_js, raw_js)
    out += u'</body></html>'

    return out


def style_single_exp_list(exp_as_list, caption):
    df = pd.DataFrame(exp_as_list)

    df_styled = df.style.background_gradient(cmap='Reds', subset=[1])

    # Add a solid line to separate the header
    df_styled.set_table_styles([
        {'selector': 'th', 'props': [('border-bottom', '2px solid black'), ('border-top', '0px')]},
        {'selector': 'td', 'props': [('border', '0px'), ('vertical-align', 'middle'), ('padding', '0.5em 0.5em')]}
    ]).format(precision=3, thousands=".", decimal=",")

    # Add a caption
    df_styled.set_caption(caption)
    return df_styled.to_html()
