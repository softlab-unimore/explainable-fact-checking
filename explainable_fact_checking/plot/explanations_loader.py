import json
import os
import pickle
import numpy as np
import pandas as pd
import explainable_fact_checking as xfc


def load_explanations_lime(dir='/homes/bussotti/feverous_work/feverousdata/AB/lime_explanations'):
    """
        Using pickle load the explanations by scanning directories in 'lime_explanations' folder
        For each folder create a dictionary where the key is the dataset_file_name and the value is a dictonary of explanations
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
    'model_params': {'model_path': 'model_path'},
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
                        tdict[subkey] = subvalue
                else:
                    tdict[key] = value
            tdict['dataset_file_name'] = tdict.pop('dataset_file')
            texp_list = explanation_to_dict_olap(explanation)
            out_list += [x | tdict for x in texp_list]
    ret_df = pd.DataFrame(out_list)
    # convert the following columns in categorical ['type', 'dataset_file_name', 'model_path', 'model_name', 'perturbation_mode', 'mode']
    for col in ['type', 'dataset_file_name', 'model_name', 'perturbation_mode', 'mode']:
        if col in ret_df.columns:
            ret_df[col] = ret_df[col].astype('category')
    # sort columns
    # first ['unit_text', 'unit_index',  'SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO',] then the rest
    columns = ['unit_text', 'unit_index', 'SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
    columns += [x for x in ret_df.columns if x not in columns]
    ret_df = ret_df[columns]
    return ret_df.apply(pd.to_numeric, errors='ignore')


def explanation_to_dict_olap(exp):
    """
    Create a dictionary from a lime explanation object
    """
    if isinstance(exp, dict):
        get_method = lambda x: exp.get(x, None)
    else:
        get_method = lambda x: getattr(exp, x) if hasattr(exp, x) else None
    assert get_method('record') is None or get_method('record').get('noisetag') is None or len(
        get_method('record')['noisetag']) == len(get_method('local_exp')[0])
    ret_list = []
    out_dict = {}
    # get available attributes of exp
    for key in ['num_samples', 'label', 'execution_time', 'id']:
        out_dict[key] = get_method(key)
    if params_to_report := get_method('params_to_report'):
        out_dict |= params_to_report
    # out_dict['local_pred'] = exp.local_pred[0]
    if 'label' not in out_dict and get_method('record') is not None:
        out_dict['label'] = get_method('record')['label']
    class_names = get_method('class_names')
    class_names = xfc.xfc_utils.class_names_load if class_names is None else class_names
    # save predict_proba of each class
    for i, tclass in enumerate(class_names):
        out_dict[tclass + '_predict_proba'] = get_method('predict_proba')[i]

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
                ret_list[i]['noisetag'] = noisetag[i]

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


def load_explanations_lime_to_df(dir='/homes/bussotti/feverous_work/feverousdata/AB/'):
    """
    Load the explanations from the lime_explanations folder and create a pandas dataframe
    """
    files_dict = load_explanations_lime(dir=dir)
    return explanations_to_df_lime(files_dict)


def load_only_claim_predictions(dir='/homes/bussotti/feverous_work/feverousdata/AB/'):
    """
    Load the predictions of the only_claim model and create a pandas dataframe

    Parameters
    ----------
    dir : str
        The directory where the predictions are stored

    Returns
    -------
    pd.DataFrame
        A pandas dataframe with the predictions of the only_claim model

    """
    files = [x for x in os.listdir(dir) if x.endswith('only_claim.json')]
    # load the files
    predictions_dict = {}
    for file in files:
        with open(os.path.join(dir, file), 'r') as f:
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
    :param experiment_code: experiment codes
    :param results_path: path to the dataset results
    :return: a pandas dataframe with the results
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


def load_preprocess_explanations(experiment_code_list: list, only_claim_exp_list: list = [], save_name=None):
    exp_object_list = []
    for experiment_code in experiment_code_list:
        x = load_experiment_result_by_code(experiment_code, xfc.experiment_definitions.C.RESULTS_DIR)
        exp_object_list += x
    explanation_df = explanations_to_df(exp_object_list)
    id_cols = ['dataset_file_name', 'id', 'model_path']

    if 'model_path' not in explanation_df.columns:
        explanation_df['model_path'] = np.NaN
    explanation_df['model_path'].replace('nan', np.NaN, inplace=True) # todo check remove
    if explanation_df['model_path'].isna().any():
        na_mask = explanation_df['model_path'].isna()
        explanation_df.loc[na_mask, 'model_path'] = './' + explanation_df.loc[na_mask, 'model_name'].astype(str)
    class_pred_cols = [x + '_predict_proba' for x in xfc.xfc_utils.class_names_load]
    index_exp = explanation_df[id_cols]

    if only_claim_exp_list:
        only_claim_list = []
        for experiment_code in only_claim_exp_list:
            only_claim = load_experiment_result_by_code(experiment_code, xfc.experiment_definitions.C.RESULTS_DIR)
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
    for tclass in xfc.xfc_utils.class_names_load:
        tmask = all_df['predicted_label'] == tclass
        all_df.loc[tmask, 'score_on_predicted_label'] = all_df.loc[tmask, tclass]

    # take last part of model_path as model_name
    all_df['model_id'] = all_df['model_path'].apply(lambda x: x.split('/')[-1])
    # save the explanations to a csv file
    # correction of label. Load dataset files by adding '_orig.jsonl' and replace the label with the one in the original file
    # load the original dataset files
    for dataset_file_name in all_df['dataset_file_name'].unique():
        dataset_file_name_orig = dataset_file_name.replace('.jsonl', '_orig.jsonl')
        tpath = os.path.join(xfc.experiment_definitions.C.DATASET_DIR_FEVEROUS[0], dataset_file_name_orig)
        if not os.path.exists(tpath):
            continue
        with open(tpath, 'r') as f:
            orig_records = [json.loads(x) for x in f]
        orig_records_dict = {x['id']: x for x in orig_records}
        mask = all_df['dataset_file_name'] == dataset_file_name
        all_df.loc[mask, 'label'] = all_df.loc[mask, 'id'].apply(lambda x: orig_records_dict[x]['label'])

    if save_name:
        all_df.to_csv(os.path.join(xfc.experiment_definitions.C.RESULTS_DIR, save_name), index=False)

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


if __name__ == '__main__':
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
