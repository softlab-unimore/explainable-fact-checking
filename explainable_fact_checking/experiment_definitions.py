import os
import sys
import explainable_fact_checking as xfc


class C:
    MODEL_DIR = 'models'
    PREDICTION_DIR = 'predictions'
    EVALUATION_DIR = 'evaluation'
    RESULTS_DIR = '/home/bussotti/XFCresults/experiments'
    PLOT_DIR = '/home/bussotti/XFCresults/plots'
    DATASET_DIR_FEVEROUS = ['/home/bussotti/XFCresults/']
    BASE_CONFIG = dict(results_dir=RESULTS_DIR, random_seed=[1], )
    feverous_datasets_conf = dict(dataset_name='feverous',
                                  dataset_params=dict(
                                      dataset_dir=DATASET_DIR_FEVEROUS,
                                      dataset_file=[
                                          'ex_AB_00.jsonl',
                                          'feverous_train_challenges_withnoise.jsonl',
                                          'original_TO_01_formatted.jsonl',
                                          'feverous_dev_ST_01.jsonl',
                                          'feverous_dev_SO_01.jsonl',
                                      ],
                                      top=[1000]),
                                  )
    JF_feverous_model = dict(model_name=['default'], model_params=dict(
        model_path=['/homes/bussotti/feverous_work/feverousdata/models_fromjf270623or']), )

    baseline_feverous_model = dict(
        model_name=['default'],
        model_params=dict(
            model_path=[
                '/homes/bussotti/feverous_work/feverousdata/modeloriginalfeverousforandrea/feverous_verdict_predictor']),
    )

    lime_only_ev = dict(explainer_name=['lime'],
                        explainer_params=dict(perturbation_mode=['only_evidence'], num_samples=[500], ),
                        )

    shap_only_ev = dict(explainer_name=['shap'],
                        explainer_params=dict(perturbation_mode=['only_evidence'], mode=['KernelExplainer'],
                                              num_samples=[500], ),
                        )

    claim_only_explainer = dict(explainer_name=['claim_only_pred'], explainer_params=dict(), )


REQUIRED_FIELDS = ['experiment_id', 'dataset_name', 'model_name', 'explainer_name',
                   ]

experiment_definitions = [
    dict(experiment_id='sk_f_jf_1.0', ) | C.BASE_CONFIG |
    C.JF_feverous_model | C.lime_only_ev | C.feverous_datasets_conf,

    dict(experiment_id='sk_f_jf_1.1', ) | C.BASE_CONFIG |
    C.JF_feverous_model | C.shap_only_ev | dict(dataset_name='feverous',
                                                dataset_params=dict(
                                                    dataset_dir=C.DATASET_DIR_FEVEROUS,
                                                    dataset_file=[
                                                        'ex_AB_00.jsonl',
                                                    ],
                                                    top=[1000]),
                                                ),
    dict(experiment_id='sk_f_jf_1.1b', ) | C.BASE_CONFIG |
    C.JF_feverous_model | C.shap_only_ev | dict(dataset_name='feverous',
                                                dataset_params=dict(
                                                    dataset_dir=C.DATASET_DIR_FEVEROUS,
                                                    dataset_file=[
                                                        'original_TO_01_formatted.jsonl',
                                                        'feverous_dev_ST_01.jsonl',
                                                        'feverous_dev_SO_01.jsonl',
                                                    ],
                                                    top=[1000]),
                                                ),
    dict(experiment_id='sk_f_jf_1.1n', ) | C.BASE_CONFIG |
    C.JF_feverous_model | C.shap_only_ev | dict(dataset_name='feverous',
                                                dataset_params=dict(
                                                    dataset_dir=C.DATASET_DIR_FEVEROUS,
                                                    dataset_file=[
                                                        'feverous_train_challenges_withnoise.jsonl',
                                                    ],
                                                    top=[1000]),
                                                ),

    dict(experiment_id='f_bs_1.0', ) | C.BASE_CONFIG |
    C.baseline_feverous_model | C.lime_only_ev | C.feverous_datasets_conf,

    dict(experiment_id='f_bs_1.1', ) | C.BASE_CONFIG |
    C.baseline_feverous_model | C.shap_only_ev | dict(dataset_name='feverous',
                                                      dataset_params=dict(
                                                          dataset_dir=C.DATASET_DIR_FEVEROUS,
                                                          dataset_file=[
                                                              'ex_AB_00.jsonl',
                                                              'feverous_train_challenges_withnoise.jsonl',
                                                          ],
                                                          top=[1000]),
                                                      ),

    dict(experiment_id='f_bs_1.1b', ) | C.BASE_CONFIG |
    C.baseline_feverous_model | C.shap_only_ev | dict(dataset_name='feverous',
                                                      dataset_params=dict(
                                                          dataset_dir=C.DATASET_DIR_FEVEROUS,
                                                          dataset_file=[
                                                              'feverous_dev_SO_01.jsonl',
                                                              'original_TO_01_formatted.jsonl',
                                                          ],
                                                          top=[1000]),
                                                      ),
    dict(experiment_id='f_bs_1.1c', ) | C.BASE_CONFIG |
    C.baseline_feverous_model | C.shap_only_ev | dict(dataset_name='feverous',
                                                      dataset_params=dict(
                                                          dataset_dir=C.DATASET_DIR_FEVEROUS,
                                                          dataset_file=["feverous_dev_ST_01.jsonl", ],
                                                          top=[1000]),
                                                      ),

    dict(experiment_id='oc_1.0', ) | C.BASE_CONFIG |
    C.JF_feverous_model | C.claim_only_explainer | C.feverous_datasets_conf,

    dict(experiment_id='oc_1.1', ) | C.BASE_CONFIG |
    C.baseline_feverous_model | C.claim_only_explainer | C.feverous_datasets_conf,

]

# 'feverous_train_challenges_withnoise.jsonl',

experiment_definitions_dict = {x['experiment_id']: x for x in experiment_definitions}
# check that there are only unique experiment ids
assert len(experiment_definitions_dict) == len(experiment_definitions), 'Experiment ids are not unique.'


def get_config_by_id(experiment_id, config_file_path=None):
    """
    Get the configuration dictionary for a specific experiment id.

    This function retrieves the configuration dictionary for a given experiment id.
    If a configuration file path is provided, the function will import the configuration
    module from that path and use the experiment definitions from there.
    Otherwise, it will use the default experiment definitions.

    Parameters
    ----------
    experiment_id : str
        The id of the experiment for which the configuration is to be retrieved.
    config_file_path : str, optional
        The path to the configuration file. If not provided, the default experiment definitions will be used.

    Returns
    -------
    dict
        The configuration dictionary for the specified experiment id.

    Raises
    ------
    ValueError
        If the experiment id is not found in the configuration file or if the experiment definitions
        are not a list or a dictionary.
    """
    if config_file_path is not None:
        config_file_path = os.path.abspath(config_file_path)
        sys.path.append(os.path.dirname(config_file_path))
        config_module = __import__(os.path.basename(config_file_path).split('.')[0])
        experiment_definitions = config_module.experiment_definitions
    else:
        experiment_definitions = xfc.experiment_definitions.experiment_definitions_dict
    exp_dict = dict()
    if isinstance(experiment_definitions, dict):
        exp_dict = experiment_definitions.get(experiment_id, None)
        if exp_dict is None:
            raise ValueError(f'Experiment id {experiment_id} not found in the configuration file.')
    elif isinstance(experiment_definitions, list):
        for x in experiment_definitions:
            if x['experiment_id'] == experiment_id:
                exp_dict = x
                break
    else:
        raise ValueError('Invalid experiment_definitions type. It must be a list or a dictionary.')
    return exp_dict
