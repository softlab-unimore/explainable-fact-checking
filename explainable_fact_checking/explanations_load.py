import json
import os
import pickle

import pandas as pd

import explainable_fact_checking as xfc


def load_explanations_lime(dir='/homes/bussotti/feverous_work/feverousdata/AB/lime_explanations'):
    """
        Using pickle load the explanations by scanning directories in 'lime_explanations' folder
        For each folder create a dictionary where the key is the filename and the value is a dictonary of explanations
        The inner dictionary has as key the name of the file and as value the explanation object.
        use scandir
    """
    exp_dir = dir
    explanations = {}
    for folder in os.scandir(exp_dir):
        if not folder.is_dir():
            continue
        explanations[folder.name] = {}
        for file in os.scandir(folder):
            if not file.name.endswith('.pkl'):
                continue
            with open(file, 'rb') as f:
                explanations[folder.name.replace('.pkl', '')][file.name] = pickle.load(f)
    return explanations


def explanations_to_df_lime(explanations_files):
    """
    Create a pandas dataframe from the explanations
    input: explanations dictionary that has as key the filename and as value a dictionary of explanations
        the second level dictionary has as key the id of the record and as value the explanation object.
    output: a pandas dataframe with the columns:
        each row represent a unit of the explanation, the columns are:
        position, text, impact, [RECORD_INFO], [EXPLANATION_INFO]
    """
    out_list = []
    for filename, explanation_dict in explanations_files.items():
        for explanation_key, explanation in explanation_dict.items():
            tdict = {}
            tdict['filename'] = filename
            tdict['id'] = explanation_key.replace('.pkl', '')
            texp_list = lime_explanation_to_dict_olap(explanation)

            out_list += [tdict | x for x in texp_list]
    ret_df = pd.DataFrame(out_list)
    # sort columns
    # first ['unit_text', 'unit_index',  'SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO',] then the rest
    columns = ['id', 'type', 'unit_text', 'unit_index', 'SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO', 'filename']
    columns += [x for x in ret_df.columns if x not in columns]
    ret_df = ret_df[columns]
    return ret_df.apply(pd.to_numeric, errors='ignore')


def lime_explanation_to_dict_olap(exp):
    """
    Create a dictionary from a lime explanation object
    """
    ret_list = []
    out_dict = {}
    for key in ['num_samples', 'label', 'execution_time']:
        if hasattr(exp, key):
            out_dict[key] = getattr(exp, key)
    out_dict['local_pred'] = exp.local_pred[0]
    if 'label' not in out_dict and hasattr(exp, 'record'):
        out_dict['label'] = exp.record['label']
    # save predict_proba of each class
    for i, tclass in enumerate(exp.class_names):
        out_dict[tclass + '_predict_proba'] = exp.predict_proba[i]

    # create a dictionary exp_by_unit which has as key the index of unit explained and as value a dictionary with the impacts on each class
    exp_by_unit = {i: {} for i in range(len(exp.local_exp[0]))}
    for i, tclass in enumerate(exp.class_names):
        tclass_exp = exp.local_exp[i]
        for unit_index, impact in tclass_exp:
            exp_by_unit[unit_index][tclass] = impact

    # for each element of the explanation create a dictionary with its impacts on each class
    for unit_index, impact_dict in exp_by_unit.items():
        tdict = impact_dict | out_dict
        tdict['unit_index'] = unit_index
        tdict['unit_text'] = exp.domain_mapper.indexed_string.inverse_vocab[unit_index]
        tdict['type'] = 'evidence'
        ret_list.append(tdict)

    # Adding the claim in unit_text and intercept in the impact
    claim_text = exp.claim if hasattr(exp, 'claim') else 'No claim available'
    claim_dict = dict(unit_index=0, unit_text=claim_text, type='claim_intercept')
    claim_dict = claim_dict | out_dict
    for i, tclass in enumerate(exp.class_names):
        claim_dict[tclass] = exp.intercept[i]
    ret_list.append(claim_dict)
    return ret_list


def load_explanations_lime_to_df(dir='/homes/bussotti/feverous_work/feverousdata/AB/'):
    """
    Load the explanations from the lime_explanations folder and create a pandas dataframe
    """
    files_dict = load_explanations_lime(dir=dir)
    return explanations_to_df_lime(files_dict)


def load_only_claim_predictions(dir='/homes/bussotti/feverous_work/feverousdata/AB/'):
    """
    Load predictions of only the claim
    :param dir:
    :return:
    """
    files = [x for x in os.listdir(dir) if x.endswith('only_claim.json')]
    # load the files
    predictions_dict = {}
    for file in files:
        with open(os.path.join(dir, file), 'r') as f:
            predictions_dict[file] = json.load(f)
    # convert the file in dataframe
    out_list = []
    for filename, predictions in predictions_dict.items():
        base_dict = {'filename': filename.replace('_pred_only_claim.json', '')}
        for record in predictions:
            t2dict = base_dict.copy()
            t2dict['unit_index'] = 0
            t2dict['unit_text'] = record['claim']
            t2dict['type'] = 'only_claim'
            key_to_copy = ['id', 'label']
            for key in key_to_copy:
                t2dict[key] = record[key]
            for i, tclass in enumerate(xfc.xfc_utils.class_names):
                t2dict[tclass] = record['predicted_scores'][i]
            out_list.append(t2dict)
    return pd.DataFrame(out_list)


def load_preprocess_explanations():
    files_dict = load_explanations_lime(dir='/homes/bussotti/feverous_work/feverousdata/AB/lime_explanations')
    id_cols = ['filename', 'id']
    explanation_df = explanations_to_df_lime(files_dict)
    index_exp = explanation_df[id_cols]

    only_claim_predictions_df = load_only_claim_predictions()
    only_claim_predictions_df.set_index(id_cols, inplace=True, drop=True)
    common_index = pd.MultiIndex.from_frame(index_exp).intersection(
        only_claim_predictions_df.index)
    only_claim_predictions_df = only_claim_predictions_df.loc[common_index].reset_index()

    class_pred_cols = [x + '_predict_proba' for x in xfc.xfc_utils.class_names]

    mask = explanation_df['type'] == 'claim_intercept'
    only_claim_predictions_df = only_claim_predictions_df.merge(explanation_df.loc[mask, id_cols + class_pred_cols])

    all_df = pd.concat([explanation_df, only_claim_predictions_df], ignore_index=True)
    # sort all_df by filename and id
    all_df.sort_values(by=id_cols, inplace=True)

    all_df['predicted_label'] = all_df[class_pred_cols].idxmax(axis=1).str.replace('_predict_proba', '')
    for tclass in xfc.xfc_utils.class_names:
        tmask = all_df['predicted_label'] == tclass
        all_df.loc[tmask, 'score_on_predicted_label'] = all_df.loc[tmask, tclass]


    # save the explanations to a csv file
    all_df.to_csv('/homes/bussotti/feverous_work/feverousdata/AB/explanations_lime.csv', index=False)

    assert (pd.Series(only_claim_predictions_df[id_cols].astype(str).apply('_'.join, 1).unique()).isin(explanation_df[
        id_cols].astype(str).apply('_'.join, 1).unique())).all(), \
        'Different (id, filename) in explanations and only_claim_predictions'
    assert explanation_df[class_pred_cols].astype(str).apply('_'.join, 1).groupby(
        explanation_df[id_cols].astype(str).apply('_'.join, 1)).nunique().max() == 1, \
        'Multiple predict_proba for the same (id, filename)'

    for filename, explanation_dict in files_dict.items():
        print(f'File: {filename}')
        print(f'Number of explanations: {len(explanation_dict)}')


if __name__ == '__main__':
    load_preprocess_explanations()
