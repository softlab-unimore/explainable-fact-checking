import os
import sys
import explainable_fact_checking as xfc


class CONSTANTS:
    MODEL_DIR = 'models'
    PREDICTION_DIR = 'predictions'
    EVALUATION_DIR = 'evaluation'
    RESULTS_DIR = '/homes/bussotti/feverous_work/feverousdata/AB/experiments'
    DATASET_DIR_FEVEROUS = ['/homes/bussotti/feverous_work/feverousdata/AB/']


REQUIRED_FIELDS = ['experiment_id', 'dataset_name', 'model_name', 'explainer_name',
                   ]

base_config = dict(
    results_dir=CONSTANTS.RESULTS_DIR,

)

experiment_definitions = [
    base_config.copy() | dict(
        experiment_id='sk_f_jf_1.0',
        dataset_name='feverous',
        dataset_params=dict(
            dataset_dir=CONSTANTS.DATASET_DIR_FEVEROUS,
            dataset_file=[
                'ex_AB_00.jsonl',
                # 'original_TO_01_formatted.jsonl',
                # 'feverous_dev_ST_01.jsonl',
                # 'feverous_dev_SO_01.jsonl',
            ],

            top=[1000]),
        model_name=['default'],
        explainer_name=['lime'],
        random_seed=[1],
        model_params=dict(pathModel=['models_fromjf270623or']),
        explainer_params=dict(perturbation_mode=['only_evidence'], mode=['KernelExplainer'], num_samples=[500], ),
    ),
    base_config.copy() | dict(
        experiment_id='sk_f_jf_1.1',
        dataset_name='feverous',
        dataset_params=dict(
            dataset_dir=CONSTANTS.DATASET_DIR_FEVEROUS,
            dataset_file=[
                'ex_AB_00.jsonl',
                # 'original_TO_01_formatted.jsonl',
                # 'feverous_dev_ST_01.jsonl',
                # 'feverous_dev_SO_01.jsonl',
            ],

            top=[1000]),
        model_name=['default'],
        explainer_name=['shap'],
        random_seed=[1],
        model_params=dict(pathModel=['models_fromjf270623or']),
        explainer_params=dict(perturbation_mode=['only_evidence'], mode=['KernelExplainer'], num_samples=[500], ),
    ),

]

experiment_definitions_dict = {x['experiment_id']: x for x in experiment_definitions}


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
