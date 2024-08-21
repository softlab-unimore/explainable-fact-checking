import numpy as np


class C:
    MODEL_DIR = 'models'
    PREDICTION_DIR = 'predictions'
    EVALUATION_DIR = 'evaluation'
    RESULTS_DIR = '/home/bussotti/XFCresults/experiments'
    PLOT_DIR = '/home/bussotti/XFCresults/plots'
    DATASET_DIR_FEVEROUS = ['/home/bussotti/XFCresults/']
    DATASET_DIR = ['/home/bussotti/XFCresults/datasets']
    POLITIHOP_DS_DIR = ['/home/bussotti/XFCresults/datasets/politihop']
    BASE_CONFIG = dict(results_dir=RESULTS_DIR, random_seed=[1], )
    ROBERTA_V2_3L = '/home/bussotti/experiment_AE/0824_explainer_newmodel/llama318b_feverousobj5trained_10epochs_3labels/adapter_config.json'

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
    feverous_ds_xs = dict(dataset_name='feverous',
                          dataset_params=dict(
                             dataset_dir=DATASET_DIR_FEVEROUS,
                             dataset_file=[
                                 'feverous_train_challenges_withnoise.jsonl',
                                 'original_TO_01_formatted.jsonl',
                                 'feverous_dev_ST_01.jsonl',
                                 'feverous_dev_SO_01.jsonl',
                             ],
                             top=[3]),
                          )
    politihop_xs_test = dict(dataset_name='politihop',
                             dataset_params=dict(
                                 dataset_dir=POLITIHOP_DS_DIR,
                                 dataset_file=[
                                     'politihop_test.tsv',
                                 ],
                                 top=[3]),
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

    not_precomputed_datasets_conf_10 = dict(dataset_name='feverous',
                                         dataset_params=dict(
                                             dataset_dir=['/home/bussotti/XFCresults/datasets'],
                                             dataset_file=[
                                                 'dev.combined.not_precomputed.p5.s5.t3_readable_test.jsonl',
                                                 'dev.combined.not_precomputed.p5.s5.t3_readable_train.jsonl',
                                                 'dev.combined.not_precomputed.p5.s20.t3_readable_test.jsonl',
                                                 'dev.combined.not_precomputed.p5.s20.t3_readable_train.jsonl',
                                             ],
                                             top=[10]),
                                         )

    fake_predictor = dict(model_name=['fake_predictor'])

    JF_feverous_model = dict(model_name=['default'], model_params=dict(
        model_path=['/homes/bussotti/feverous_work/feverousdata/models_fromjf270623or']), )

    baseline_feverous_model = dict(
        model_name=['default'],
        model_params=dict(
            model_path=[
                '/homes/bussotti/feverous_work/feverousdata/modeloriginalfeverousforandrea/feverous_verdict_predictor']),
    )

    roberta_v0 = dict(
        model_name=['Roberta'],
        model_params=dict(
            base_model_name="meta-llama/Meta-Llama-3.1-8B",
        ),
    )

    lime_only_ev = dict(explainer_name=['lime'],
                        explainer_params=dict(perturbation_mode=['only_evidence'], num_samples=[500], ),
                        )

    lime_only_ev_50 = dict(explainer_name=['lime'],
                           explainer_params=dict(perturbation_mode=['only_evidence'], num_samples=[50], ),
                           )

    shap_only_ev = dict(explainer_name=['shap'],
                        explainer_params=dict(perturbation_mode=['only_evidence'], mode=['KernelExplainer'],
                                              num_samples=[500], ),
                        )
    shap_only_ev_50 = dict(explainer_name=['shap'],
                           explainer_params=dict(perturbation_mode=['only_evidence'], mode=['KernelExplainer'],
                                                 num_samples=[50], ),
                           )
    n_perturb_time = [int(x) for x in (2 ** np.arange(5, 13 + 1))]
    lime_only_ev_time = dict(explainer_name=['lime'],
                             explainer_params=dict(perturbation_mode=['only_evidence'], num_samples=n_perturb_time, ),
                             )
    lime_only_ev_time_v2_s = dict(explainer_name=['lime'],
                                  explainer_params=dict(perturbation_mode=['only_evidence'],
                                                        num_samples=n_perturb_time[:3], ),
                                  )
    shap_only_ev_time = dict(explainer_name=['shap'],
                             explainer_params=dict(perturbation_mode=['only_evidence'], mode=['KernelExplainer'],
                                                   num_samples=n_perturb_time, ),
                             )
    shap_only_ev_time_v2_s = dict(explainer_name=['shap'],
                                  explainer_params=dict(perturbation_mode=['only_evidence'], mode=['KernelExplainer'],
                                                        num_samples=n_perturb_time[:3], ),
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

    dict(experiment_id='lla_np_1.0', ) | C.BASE_CONFIG |
    C.roberta_v0 | C.lime_only_ev | C.not_precomputed_datasets_conf,

    dict(experiment_id='lla_np_2.0', ) | C.BASE_CONFIG |
    C.roberta_v0 | C.shap_only_ev | C.not_precomputed_datasets_conf,

    dict(experiment_id='lla_np_1.test', ) | C.BASE_CONFIG |
    C.roberta_v0 | C.lime_only_ev | C.not_precomputed_datasets_conf_10,

    dict(experiment_id='oc_fbs_np_1.0', ) | C.BASE_CONFIG |
    C.baseline_feverous_model | C.claim_only_explainer | C.not_precomputed_datasets_conf,

    # time scalability
    dict(experiment_id='fbs_time_1.0', ) | C.BASE_CONFIG |
    C.baseline_feverous_model | C.lime_only_ev_time | C.feverous_ds_100,

    dict(experiment_id='fbs_time_2.0', ) | C.BASE_CONFIG |
    C.baseline_feverous_model | C.shap_only_ev_time | C.feverous_ds_100,

    dict(experiment_id='fbs_time_1.1', ) | C.BASE_CONFIG |
    C.baseline_feverous_model | C.lime_only_ev_time_v2_s | C.feverous_ds_100,

    dict(experiment_id='fbs_time_2.1', ) | C.BASE_CONFIG |
    C.baseline_feverous_model | C.shap_only_ev_time_v2_s | C.feverous_ds_100,

    dict(experiment_id='fbs_time_2.1test', ) | C.BASE_CONFIG |
    C.baseline_feverous_model | C.shap_only_ev_time_v2_s | C.feverous_ds_100,
    # after having the results of the time experiment
    # define the best number of samples experiment with less combinations of num_samples.

    dict(experiment_id='sms_p_1.0', ) | C.BASE_CONFIG |
    C.fake_predictor |
    C.lime_only_ev_50
    # dict(explainer_name=['lime'],
    #      explainer_params=dict(perturbation_mode=['only_evidence'], num_samples=[50], separator=r'<|reserved_special_token_15|>', ),
    #      )
    | C.feverous_ds_100
    # C.politihop_10_test
    ,

    dict(experiment_id='test_3.0', ) | C.BASE_CONFIG | C.roberta_v0 | C.lime_only_ev_50 | C.feverous_ds_xs,
    dict(experiment_id='test_3.1', ) | C.BASE_CONFIG | C.roberta_v0 | C.shap_only_ev_50 | C.feverous_ds_xs,
    dict(experiment_id='test_2.0', ) | C.BASE_CONFIG | C.baseline_feverous_model | C.lime_only_ev_50 | C.feverous_ds_xs,
    dict(experiment_id='test_2.1', ) | C.BASE_CONFIG | C.baseline_feverous_model | C.shap_only_ev_50 | C.feverous_ds_xs,
    dict(experiment_id='test_1.0', ) | C.BASE_CONFIG | C.fake_predictor | C.lime_only_ev_50 | C.politihop_xs_test,
    dict(experiment_id='test_1.1', ) | C.BASE_CONFIG | C.fake_predictor | C.shap_only_ev_50 | C.politihop_xs_test,

]
