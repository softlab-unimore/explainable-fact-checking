import os
import sys

import numpy as np

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
    feverous_ds_100 = dict(dataset_name='feverous',
                           dataset_params=dict(
                               dataset_dir=DATASET_DIR_FEVEROUS,
                               dataset_file=[
                                   'feverous_train_challenges_withnoise.jsonl',
                                   'original_TO_01_formatted.jsonl',
                                   'feverous_dev_ST_01.jsonl',
                                   'feverous_dev_SO_01.jsonl',
                               ],
                               top=[100]),
                           )
    not_precomputed_datasets_conf = dict(dataset_name='feverous',
                                         dataset_params=dict(
                                             dataset_dir=['/home/bussotti/XFCresults/datasets'],
                                             dataset_file=[
                                                 'dev.combined.not_precomputed.p5.s5.t3_readable_test.jsonl',
                                                 'dev.combined.not_precomputed.p5.s5.t3_readable_train.jsonl',
                                                 'dev.combined.not_precomputed.p5.s20.t3_readable_test.jsonl',
                                                 'dev.combined.not_precomputed.p5.s20.t3_readable_train.jsonl',
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
    n_perturb_time = [int(x) for x in (2 ** np.arange(5,13+1))]
    lime_only_ev_time = dict(explainer_name=['lime'],
                             explainer_params=dict(perturbation_mode=['only_evidence'], num_samples=n_perturb_time, ),
                             )
    lime_only_ev_time_v2_s = dict(explainer_name=['lime'],
                             explainer_params=dict(perturbation_mode=['only_evidence'], num_samples=n_perturb_time[:3], ),
                             )
    shap_only_ev_time = dict(explainer_name=['shap'],
                        explainer_params=dict(perturbation_mode=['only_evidence'], mode=['KernelExplainer'],
                                              num_samples=n_perturb_time, ),
                        )

    claim_only_explainer = dict(explainer_name=['claim_only_pred'], explainer_params=dict(), )


REQUIRED_FIELDS = ['experiment_id', 'dataset_name', 'model_name', 'explainer_name',
                   ]

experiment_definitions_list = [
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
    dict(experiment_id='f_bs_1.0test', ) | C.BASE_CONFIG |
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

    # retrived datasets not_precomputed
    dict(experiment_id='fbs_np_1.0', ) | C.BASE_CONFIG |
    C.baseline_feverous_model | C.lime_only_ev | C.not_precomputed_datasets_conf,

    dict(experiment_id='fbs_np_2.0', ) | C.BASE_CONFIG |
    C.baseline_feverous_model | C.shap_only_ev | C.not_precomputed_datasets_conf,

    dict(experiment_id='oc_fbs_np_1.0', ) | C.BASE_CONFIG |
    C.baseline_feverous_model | C.claim_only_explainer | C.not_precomputed_datasets_conf,

    # time scalability
    dict(experiment_id='fbs_time_1.0', ) | C.BASE_CONFIG |
    C.baseline_feverous_model | C.lime_only_ev_time | C.feverous_ds_100,

    dict(experiment_id='fbs_time_2.0', ) | C.BASE_CONFIG |
    C.baseline_feverous_model | C.shap_only_ev_time | C.feverous_ds_100,

    dict(experiment_id='fbs_time_1.1', ) | C.BASE_CONFIG |
    C.baseline_feverous_model | C.lime_only_ev_time_v2_s | C.feverous_ds_100,
    # after having the results of the time experiment
    # define the best number of samples experiment with less combinations of num_samples.

]


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
        experiment_definitions_list = config_module.experiment_definitions_list
    else:
        experiment_definitions_list = xfc.experiment_definitions.experiment_definitions_list

    if isinstance(experiment_definitions_list, list):
        experiment_definitions_dict = {x['experiment_id']: x for x in experiment_definitions_list}
        # check that there are only unique experiment ids
        assert len(experiment_definitions_dict) == len(experiment_definitions_list), 'Experiment ids are not unique.'
    else:
        raise ValueError('Invalid experiment_definitions type. It must be a list or a dictionary.')

    exp_dict = experiment_definitions_dict.get(experiment_id, None)
    if exp_dict is None:
        raise ValueError(f'Experiment id {experiment_id} not found in the configuration file.')

    return exp_dict
