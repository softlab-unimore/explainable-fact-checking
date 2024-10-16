import copy
import json
import os
import pickle
import numpy as np
import pandas as pd
import explainable_fact_checking as xfc
import itertools as it


def load_explanations_lime(path='/homes/bussotti/feverous_work/feverousdata/AB/lime_explanations'):
    """
        Using pickle load the explanations by scanning directories in 'lime_explanations' folder
        For each folder create a dictionary where the key is the dataset_file_name and the value is a dictonary of explanations
        The inner dictionary has as key the name of the file and as value the explanation object.
        use scandir
    """
    exp_dir = path
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
    input: explanations dictionary that has as key the dataset_file_name and as value a dictionary of explanations
        the second level dictionary has as key the id of the record and as value the explanation object.
    output: a pandas dataframe with the columns:
        each row represent a unit of the explanation, the columns are:
        position, text, impact, [RECORD_INFO], [EXPLANATION_INFO]
    """
    out_list = []
    for dataset_file_name, explanation_dict in explanations_files.items():
        for explanation_key, explanation in explanation_dict.items():
            tdict = {}
            texp_list = explanation_to_dict_olap(explanation)

            out_list += [tdict | x for x in texp_list]
    ret_df = pd.DataFrame(out_list)
    # sort columns
    # first ['unit_text', 'unit_index',  'SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO',] then the rest
    columns = ['id', 'type', 'unit_text', 'unit_index', 'SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO', 'dataset_file_name']
    columns += [x for x in ret_df.columns if x not in columns]
    ret_df = ret_df[columns]
    return ret_df.apply(pd.to_numeric, errors='ignore')


def explanations_to_df(explanation_object_list: list):
    """
    Create a pandas dataframe from the explanations
    input: explanations list, where each element is a pair of (params_dict, explanation_list)
    e.g. of pair: ({'experiment_id': 'sk_f_jf_1.0', 'dataset_name': 'feverous', 'random_seed': 1, 'model_name': 'default',
    'model_params': {mpath: mpath},
     'explainer_name': 'lime', 'dataset_params': {'dataset_dir': 'dataset_dir',
            'dataset_file_name': 'dataset_name.jsonl', 'top': 1000},
            'explainer_params': {'perturbation_mode': 'only_evidence', 'num_samples': 500}},
            [exp1, exp2, ...])
    output: a pandas dataframe with the columns:
        each row represent a unit of the explanation, and all values of the params_dict and the explanation_list are added as columns.

    """
    out_list = []
    for params_dict, explanation_list in explanation_object_list:
        for explanation in explanation_list:
            # tdict = params_dict.copy()
            # the params dict has sub dictionaries, we need to flatten it
            tdict = {}
            for key, value in params_dict.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        tdict[key + '__' + subkey] = subvalue
                else:
                    tdict[key] = value
            if 'dataset_params__dataset_file' in tdict:
                tdict['dataset_file_name'] = tdict.pop('dataset_params__dataset_file')
            else:
                tdict['dataset_file_name'] = tdict['dataset_name']
            if 'explainer_params__class_names' in tdict:
                tdict['explainer_params__class_names'] = [xfc.C.CLASS_MAP.get(x, x) for x in
                                                          tdict['explainer_params__class_names']]
            else:
                tdict['explainer_params__class_names'] = xfc.experiment_definitions.E.CLASS_NAMES_V0
            texp_list = explanation_to_dict_olap(explanation, tdict)
            out_list += [x | tdict for x in texp_list]
    ret_df = pd.DataFrame(out_list)
    # convert the following columns in categorical ['type', 'dataset_file_name', mpath, 'model_name', 'perturbation_mode', 'mode']
    for col in ['type', 'dataset_file_name', 'model_name', 'perturbation_mode', 'mode']:
        if col in ret_df.columns:
            ret_df[col] = ret_df[col].astype('category')
    for col in ['goldtag']:
        if col in ret_df.columns:
            ret_df[col] = ret_df[col].astype('Int64')
    # sort columns
    # first ['unit_text', 'unit_index',  'SUPPORTS', 'REFUTES'] then the rest
    unique_cnames = set(it.chain.from_iterable(ret_df['explainer_params__class_names'].values))
    unique_cnames = set([xfc.C.CLASS_MAP.get(x, x) for x in unique_cnames])
    columns = ['unit_text', 'unit_index', 'SUPPORTS', 'REFUTES', 'NEI']
    columns += [x for x in xfc.xfc_utils.ordered_set(it.chain(unique_cnames, ret_df.columns)) if x not in columns]
    ret_df = ret_df[columns]
    mpath = 'model_params__model_path'
    if mpath not in ret_df.columns:
        ret_df[mpath] = np.NaN
    ret_df[mpath].replace('nan', np.NaN, inplace=True)  # todo check remove
    if ret_df[mpath].isna().any():
        na_mask = ret_df[mpath].isna()
        ret_df.loc[na_mask, mpath] = './' + ret_df.loc[na_mask, 'model_name'].astype(str)
    return ret_df.apply(pd.to_numeric, errors='ignore')


def explanation_to_dict_olap(exp, prm_dict=None):
    """
    Create a dictionary from a lime explanation object
    """
    if isinstance(exp, dict):
        get_method = lambda x: exp.get(x, None)
    else:
        get_method = lambda x: getattr(exp, x) if hasattr(exp, x) else None
    assert get_method('record') is None or get_method('record').get('noisetag') is None or len(
        get_method('record')['noisetag']) == len(get_method('local_exp')[0])
    assert (get_method('record') is None or get_method('record').get('goldtag') is None or
            len(get_method('record')['goldtag']) == len(get_method('local_exp')[0]))
    ret_list = []
    out_dict = {}
    # get available attributes of exp
    for key in ['label', 'execution_time', 'id']:
        out_dict[key] = get_method(key)
    if params_to_report := get_method('params_to_report'):
        out_dict |= params_to_report
    # out_dict['local_pred'] = exp.local_pred[0]
    if 'label' not in out_dict and get_method('record') is not None:
        out_dict['label'] = get_method('record')['label']
    class_names = get_method('class_names')
    if class_names is None:
        class_names = prm_dict.get('explainer_params__class_names', None)
        if class_names is None:
            class_names = xfc.experiment_definitions.E.CLASS_NAMES_V1
    class_names = [xfc.C.CLASS_MAP.get(x, x) for x in class_names]
    # save predict_proba of each class
    predict_proba = get_method('predict_proba')
    for i, tclass in enumerate(class_names):
        out_dict[tclass + '_predict_proba'] = predict_proba[i]

    # create a dictionary exp_by_unit which has as key the index of unit explained and as value a dictionary with the impacts on each class
    local_exp = get_method('local_exp')
    if local_exp is not None:
        exp_by_unit = {i: {} for i in range(len(local_exp[0]))}
        for i, tclass in enumerate(class_names):
            tclass_exp = local_exp[i]
            for unit_index, impact in tclass_exp:
                exp_by_unit[unit_index][tclass] = impact

        # for each element of the explanation create a dictionary with its impacts on each class
        domain_mapper = get_method('domain_mapper')
        for unit_index, impact_dict in exp_by_unit.items():
            tdict = impact_dict | out_dict
            tdict['unit_index'] = unit_index
            tdict['unit_text'] = domain_mapper.indexed_string.inverse_vocab[unit_index]
            tdict['type'] = 'evidence'
            ret_list.append(tdict)

        # Adding the noisetag if available for noisy evidence
        noisetag = get_method('record').get('noisetag', None)
        if noisetag is not None:
            for i in range(len(ret_list)):
                ret_list[i]['goldtag'] = 1 - noisetag[i]
        goldtag = get_method('record').get('goldtag', None)
        if goldtag is not None:
            for i in range(len(ret_list)):
                ret_list[i]['goldtag'] = goldtag[i]

    # Adding the claim in unit_text and intercept in the impact
    claim_text = get_method('claim')
    claim_text = 'No claim available' if claim_text is None else claim_text
    claim_dict = dict(unit_index=0, unit_text=claim_text, type='claim_intercept')
    claim_dict = claim_dict | out_dict
    intercept = get_method('intercept')
    for i, tclass in enumerate(class_names):
        claim_dict[tclass] = intercept[i]
    ret_list.append(claim_dict)
    return ret_list


def load_explanations_lime_to_df(path='/homes/bussotti/feverous_work/feverousdata/AB/'):
    """
    Load the explanations from the lime_explanations folder and create a pandas dataframe
    """
    files_dict = load_explanations_lime(path=path)
    return explanations_to_df_lime(files_dict)


def load_only_claim_predictions(path='/homes/bussotti/feverous_work/feverousdata/AB/'):
    """
    Load the predictions of the only_claim model and create a pandas dataframe

    Parameters
    ----------
    path : str
        The directory where the predictions are stored

    Returns
    -------
    pd.DataFrame
        A pandas dataframe with the predictions of the only_claim model

    """
    files = [x for x in os.listdir(path) if x.endswith('only_claim.json')]
    # load the files
    predictions_dict = {}
    for file in files:
        with open(os.path.join(path, file), 'r') as f:
            predictions_dict[file] = json.load(f)
    # convert the file in dataframe
    out_list = []
    for dataset_file_name, predictions in predictions_dict.items():
        base_dict = {'dataset_file_name': dataset_file_name.replace('_pred_only_claim.json', '')}
        for record in predictions:
            t2dict = base_dict.copy()
            t2dict['unit_index'] = 0
            t2dict['unit_text'] = record['claim']
            t2dict['type'] = 'only_claim'
            key_to_copy = ['id', 'label']
            for key in key_to_copy:
                t2dict[key] = record[key]
            for i, tclass in enumerate(xfc.xfc_utils.class_names_load):
                t2dict[tclass] = record['predicted_scores'][i]
            out_list.append(t2dict)
    return pd.DataFrame(out_list)


def load_experiment_result_by_code(experiment_code, results_path) -> list:
    """
    Load the results of the experiments given the experiment code

    Parameters
    ----------
    experiment_code : str
        experiment codes
    results_path : str
        path to the dataset results

    Returns
    -------
    list
        a list of results containing in each item a pair of params and the results


    """
    full_experiment_path = os.path.join(results_path, experiment_code)
    # scan the directory. In each subfolder there is a json file with the params and a pkl file with the results
    # create a list of results containing in each item a pair of params and the results
    results_list = []
    for folder in os.scandir(full_experiment_path):
        if not folder.is_dir():
            continue
        if 'old' in folder.name:
            continue
        params, results = None, None
        for file in os.scandir(folder):
            if file.name.endswith('.json'):
                with open(file, 'r') as f:
                    params = json.load(f)
            elif file.name.endswith('.pkl'):
                with open(file, 'rb') as f:
                    results = pickle.load(f)
        results_list.append((params, results))
    return results_list


def load_preprocess_explanations(experiment_code_list: list, only_claim_exp_list=None, save_name=None,
                                 results_path=None):
    if results_path is None:
        results_path = xfc.experiment_definitions.E.RESULTS_DIR
    if only_claim_exp_list is None:
        only_claim_exp_list = []
    exp_object_list = []
    for experiment_code in experiment_code_list:
        x = load_experiment_result_by_code(experiment_code, results_path=results_path)
        exp_object_list += x
    explanation_df = explanations_to_df(exp_object_list)
    mpath = 'model_params__model_path'
    id_cols = ['dataset_file_name', 'id', mpath]
    unique_cnames = set(it.chain.from_iterable(explanation_df['explainer_params__class_names'].values))
    unique_cnames = set([xfc.C.CLASS_MAP.get(x, x) for x in unique_cnames])
    class_pred_cols = [x + '_predict_proba' for x in unique_cnames]
    index_exp = explanation_df[id_cols]

    if only_claim_exp_list:
        only_claim_list = []
        for experiment_code in only_claim_exp_list:
            only_claim = load_experiment_result_by_code(experiment_code, results_path=results_path)
            only_claim_list += only_claim

        only_claim_predictions_df = explanations_to_df(only_claim_list)
        only_claim_predictions_df['type'] = 'only_claim'

        only_claim_predictions_df.set_index(id_cols, inplace=True, drop=True)
        common_index = pd.MultiIndex.from_frame(index_exp).intersection(
            only_claim_predictions_df.index)
        only_claim_predictions_df = only_claim_predictions_df.loc[common_index].reset_index().drop(
            columns=class_pred_cols)
        mask = explanation_df['type'] == 'claim_intercept'
        only_claim_predictions_df = only_claim_predictions_df.merge(explanation_df.loc[mask, id_cols + class_pred_cols],
                                                                    on=id_cols)
        all_df = pd.concat([explanation_df, only_claim_predictions_df], ignore_index=True)
    else:
        all_df = explanation_df

    # sort all_df by dataset_file_name and id
    all_df.sort_values(by=id_cols, inplace=True)

    all_df['predicted_label'] = all_df[class_pred_cols].idxmax(axis=1).str.replace('_predict_proba', '')
    all_df['predicted_label_int'] = all_df.apply(
        lambda xin: xin['explainer_params__class_names'].index(xin['predicted_label']), axis=1)
    for tclass in xfc.xfc_utils.class_names_load:
        tmask = all_df['predicted_label'] == tclass
        all_df.loc[tmask, 'score_on_predicted_label'] = all_df.loc[tmask, tclass]

    # take last part of model_path as model_name
    all_df['model_id'] = all_df[mpath].apply(lambda xin: xin.split('/')[-1])
    # save the explanations to a csv file
    # correction of label. Load dataset files by adding '_orig.jsonl' and replace the label with the one in the original file
    # load the original dataset files
    for dataset_file_name in all_df['dataset_file_name'].unique():
        dataset_file_name_orig = dataset_file_name.replace('.jsonl', '_orig.jsonl')
        tpath = os.path.join(xfc.experiment_definitions.E.BASE_V1[0], dataset_file_name_orig)
        if not os.path.exists(tpath):
            continue
        with open(tpath, 'r') as f:
            orig_records = [json.loads(x) for x in f]
        orig_records_dict = {x['id']: x for x in orig_records}
        mask = all_df['dataset_file_name'] == dataset_file_name
        all_df.loc[mask, 'label'] = all_df.loc[mask, 'id'].apply(lambda xin: orig_records_dict[xin]['label'])

    if save_name:
        all_df.to_csv(os.path.join(results_path, save_name), index=False)

    # assert (pd.Series(only_claim_predictions_df[id_cols].astype(str).apply('_'.join, 1).unique()).isin(explanation_df[
    #                                                                                                        id_cols].astype(
    #     str).apply('_'.join, 1).unique())).all(), \
    #     'Different (id, dataset_file_name) in explanations and only_claim_predictions'
    # assert explanation_df[class_pred_cols].astype(str).apply('_'.join, 1).groupby(
    #     explanation_df[id_cols].astype(str).apply('_'.join, 1)).nunique().max() == 1, \
    #     'Multiple predict_proba for the same (id, dataset_file_name)'
    #
    # for dataset_file_name, explanation_dict in files_dict.items():
    #     print(f'File: {dataset_file_name}')
    #     print(f'Number of explanations: {len(explanation_dict)}')

    return all_df


swapped_experiments = ['fv_f2l_1.0', 'fv_f2l_2.0', 'fv_f3l_1.0', 'fv_f3l_2.0', 'fv_f2lF_1.0', 'fv_f2lF_2.0',
                       'fv_f3lF_1.0', 'fv_f3lF_2.0', 'fv_sf_1.0', 'fv_sf_2.0', 'fv_fm_1.0', 'fv_fm_2.0', 'fv_av_1.0',
                       'fv_av_2.0', 'r2_fv_f2l_1.0', 'r2_fv_f2l_2.0', 'r2_fv_f3l_1.0', 'r2_fv_f3l_2.0',
                       'r2_fv_f2lF_1.0', 'r2_fv_f2lF_2.0', 'r2_fv_f3lF_1.0', 'r2_fv_f3lF_2.0', 'r2_fv_sf_1.0',
                       'r2_fv_sf_2.0', 'r2_fv_fm_1.0', 'r2_fv_fm_2.0',
                       ]

if __name__ == '__main__':
    df = load_preprocess_explanations(experiment_code_list=[
        # 'fv_sf_1.0', # old
        # 'fv_sf_2.0', # old
        'r2_fv_sf_1.0',
        'r2_fv_sf_2.0',

        'fv_f2lF_1.0',
        'fv_f2lF_2.0',
        'fv_f3lF_1.0',
        'fv_f3lF_2.0',

        # 'r2_fv_f2lF_1.0', # no noise
        # 'r2_fv_f2lF_2.0', # no noise
        # 'r2_fv_f3lF_1.0', # no noise
        # 'r2_fv_f3lF_2.0', # no noise

        # 'fv_fm_1.0', # v1
        # 'fv_fm_2.0', # v1
        'r2_fv_fm_1.0',
        'r2_fv_fm_2.0',

        'fv_av_1.0',
        'fv_av_2.0',

        # 'r2_fv_f2l_1.0', #f2l small
        # 'r2_fv_f2l_2.0', #f2l small
        # 'r2_fv_f3l_1.0', #f2l small
        # 'r2_fv_f3l_2.0', #f2l small
        # 'fv_f2l_1.0', #f2l small
        # 'fv_f2l_2.0', #f2l small
        # 'fv_f3l_1.0', #f2l small
        # 'fv_f3l_2.0', #f2l small
        # 'gfce_f2l_1.0', #f2l small
        # 'gfce_f2l_2.0', #f2l small

        # 'gfce_f3l_1.0',  # OK 4:27h # v1 remove
        # 'gfce_fm2_1.0',  # OK 4:29h # v1
        # 'gfce_sf_1.0',  # OK 1:10h # v1 remove
        # 'gfce_f3l_1.1',  # 4:49h # xs num_samples
        'gfce_sf_2.1',  # 5:34h
        'gfce_sf_1.1',  # 1:10h # xs num_samples
        'gfce_f3l_2.1',  # 22:52h
        # 'gfce_f3l_2.0',  # 22:25h # v1 remove
        'gfce_f3l_1.1F',  # 12:17h
        'gfce_sf_1.1F',  # 3:09h
        'gfce_av_1.0F',  # 8:00h
        'gfce_av_2.0',  # 13:55h
        # 'gfce_fm2_1.1',  # 22:28h
        # 'gfce_fm2_2.1',  # 15:39h
        # 'gfce_f2l_1.1',  # 12:31h # small f2l
        # 'gfce_f2l_2.1',  # 29:51h # small f2l
        'gfce_f2l_1.1F',  # 12:24h
        'gfce_f2l_2.1F',  # 23:08h
        'gfce_fm2_2.2',
        'gfce_fm2_1.2',

        # 'llama70b_f2l_1.0',  # 10:43h

        'llama70b_f2l_2.1',  # 14:58h
        'llama70b_fm2_1.0',  # 18:40h
        'llama70b_fm2_2.0',
        'llama70b_sf_1.0',
        'llama70b_sf_2.0',
        'llama70b_av_1.0',
        'llama70b_av_2.0',
        'llama70b_f3l_1.0',
        'llama70b_f3l_2.0',  # 17:45h
        'llama70b_f2l_1.1',  # 8:17h
    ], save_name='all_exp_v1.csv')
    model_name_map = {'Roberta_v2': 'Roberta',
                      'GenFCExp_v2': 'GenFCExp'}
    df['model_name'] = df['model_name'].replace(model_name_map)
    df.to_csv(os.path.join(xfc.experiment_definitions.E.RESULTS_DIR, 'all_exp_v2.csv'), index=False)

    df = load_preprocess_explanations(experiment_code_list=[
        'fbs_np_1.0',
        'fbs_np_2.0',
        'lla_np_1.0',
        'lla_np_2.0',
    ])
    load_preprocess_explanations(experiment_code_list=[
        'sk_f_jf_1.1',
        'sk_f_jf_1.0',
        'sk_f_jf_1.1b',
        'sk_f_jf_1.1n',
        'f_bs_1.0',
        'f_bs_1.1',
        'f_bs_1.1b',
        'f_bs_1.1c',
        'fbs_np_1.0',
        'fbs_np_2.0',
    ], only_claim_exp_list=[
        'oc_1.0',
        'oc_1.1',
        'oc_fbs_np_1.0',
    ], save_name='all_exp.csv')
