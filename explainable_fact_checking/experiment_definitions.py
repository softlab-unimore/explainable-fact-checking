import copy
from pydantic.utils import deep_update as du
from os.path import join as ospj
import numpy as np
import json


# save the experiment_definitions_list in a file
def check_experiments(experiment_definitions_list, required_fields=None):
    # with open('experiment_definitions_list2.json', 'w') as f:
    #     json.dump(experiment_definitions_list, f, indent=2)

    # check that dataset_names and model_names are registered in the factories.
    from explainable_fact_checking.datasets_loaders import dataset_loader_factory
    from explainable_fact_checking.models import model_factory
    from explainable_fact_checking.explainers import explainer_factory

    models_names = model_factory.get_available_keys()
    datasets_names = dataset_loader_factory.get_available_keys()
    explainers_names = explainer_factory.get_available_keys()

    for exp_def in experiment_definitions_list:
        if required_fields is not None:
            for rf in required_fields:
                f_list = rf.split('.')
                tdict = exp_def
                for trf in f_list:
                    if trf not in tdict:
                        raise ValueError(f"required field {rf} not found in experiment definition {exp_def}")
                    tdict = tdict[trf]

        tnames = exp_def['dataset_name']
        tnames = [tnames] if isinstance(tnames, str) else tnames
        for t in tnames:
            if t not in datasets_names:
                raise ValueError(f"dataset_name {t} not in the available datasets."
                                 f"Available datasets are {datasets_names}."
                                 f"To add a new dataset, register it in the dataset_loader_factory in datasets_loaders.py")
        tnames = exp_def['model_name']
        tnames = [tnames] if isinstance(tnames, str) else tnames
        for t in tnames:
            if t not in models_names:
                raise ValueError(f"model_name {t} not in the available models."
                                 f"Available models are {models_names}."
                                 f"To add a new model, register it in the model_factory in models.py")
        tnames = exp_def['explainer_name']
        tnames = [tnames] if isinstance(tnames, str) else tnames
        for t in tnames:
            if t not in explainers_names:
                raise ValueError(f"explainer_name {t} not in the available explainers."
                                 f"Available explainers are {explainers_names}."
                                 f"To add a new explainer, register it in the explainer_factory in explainers.py")


CLASS_NAMES_V0 = ('NEI', 'SUPPORTS', 'REFUTES')


class E:  # Experiment constants
    MODEL_DIR = 'models'
    PREDICTION_DIR = 'predictions'
    EVALUATION_DIR = 'evaluation'
    BASE_DIR_V2 = '/home/bussotti/experiments_Andrea_JF_ShapAndCo/'
    MODELS_DIR_V2 = ospj(BASE_DIR_V2, 'models')
    RESULTS_DIR = '/home/bussotti/XFCresults/experiments'
    PLOT_DIR = '/home/bussotti/XFCresults/plots'
    BASE_V1 = '/home/bussotti/XFCresults/'
    DATASET_DIR = '/home/bussotti/XFCresults/datasets'
    DATASET_DIR_V2 = BASE_DIR_V2 + 'datasets'
    DATASET_DIR_V3 = BASE_DIR_V2 + 'datasets/data'
    POLITIHOP_DS_DIR = '/home/bussotti/XFCresults/datasets/politihop'
    LIARPLUS_DS_DIR = '/home/bussotti/XFCresults/datasets/LIARPLUS'
    BASE_CONFIG = dict(results_dir=RESULTS_DIR, random_seed=[1], )
    ROBERTA_V2_3L = '/home/bussotti/experiment_AE/0824_explainer_newmodel/llama318b_feverousobj5trained_10epochs_3labels/adapter_config.json'
    LLAMA_CACHE = ospj(BASE_V1, 'llama_cache')

    RANDOM_SEEDS_v1 = dict(random_seed=[2, 3, 4, 5, 6])
    CLASS_NAMES_V1 = ('REFUTES', 'NEI', 'SUPPORTS')
    CLASS_NAMES_2L_V1 = ('REFUTES', 'SUPPORTS')
    CLASS_NAMES_V0 = CLASS_NAMES_V0
    EXPLAINER_CNAMES_3L_V1 = dict(explainer_params=dict(class_names=CLASS_NAMES_V1))
    EXPLAINER_CNAMES_2L_V1 = dict(explainer_params=dict(class_names=CLASS_NAMES_2L_V1))
    EXP_CLASS_NAMES_3L_V0 = dict(explainer_params=dict(class_names=CLASS_NAMES_V0))
    AV_CLASS_NAMES = dict(explainer_params=dict(class_names=("Refuted", "Not Enough Evidence", "Supported",
                                                             "Conflicting Evidence/Cherrypicking")))

    feverous_datasets_conf = dict(dataset_name='feverous',
                                  dataset_params=dict(
                                      dataset_dir=BASE_V1,
                                      dataset_file=[
                                          'ex_AB_00.jsonl',
                                          'feverous_train_challenges_withnoise.jsonl',
                                          'original_TO_01_formatted.jsonl',
                                          'feverous_dev_ST_01.jsonl',
                                          'feverous_dev_SO_01.jsonl',
                                      ],
                                      nrows=[1000]),
                                  )
    feverous_datasets_conf_100 = dict(dataset_name='feverous',
                                      dataset_params=dict(
                                          dataset_dir=BASE_V1,
                                          dataset_file=[
                                              'ex_AB_00.jsonl',
                                              'feverous_train_challenges_withnoise.jsonl',
                                              'original_TO_01_formatted.jsonl',
                                              'feverous_dev_ST_01.jsonl',
                                              'feverous_dev_SO_01.jsonl',
                                          ],
                                          nrows=[100]),
                                      )
    feverous_ds_100 = dict(dataset_name='feverous',
                           dataset_params=dict(
                               dataset_dir=BASE_V1,
                               dataset_file=[
                                   'feverous_train_challenges_withnoise.jsonl',
                                   'original_TO_01_formatted.jsonl',
                                   'feverous_dev_ST_01.jsonl',
                                   'feverous_dev_SO_01.jsonl',
                               ],
                               nrows=[100]),
                           )
    feverous_ds_xs = dict(dataset_name='feverous',
                          dataset_params=dict(
                              dataset_dir=BASE_V1,
                              dataset_file=[
                                  'feverous_train_challenges_withnoise.jsonl',
                                  'original_TO_01_formatted.jsonl',
                                  'feverous_dev_ST_01.jsonl',
                                  'feverous_dev_SO_01.jsonl',
                              ],
                              nrows=[3]),
                          )

    not_precomputed_datasets_conf = dict(dataset_name='feverous',
                                         dataset_params=dict(
                                             dataset_dir=DATASET_DIR,
                                             dataset_file=[
                                                 'dev.combined.not_precomputed.p5.s5.t3_readable_test.jsonl',
                                                 'dev.combined.not_precomputed.p5.s5.t3_readable_train.jsonl',
                                                 'dev.combined.not_precomputed.p5.s20.t3_readable_test.jsonl',
                                                 'dev.combined.not_precomputed.p5.s20.t3_readable_train.jsonl',
                                             ],
                                             nrows=[1000]),
                                         )
    not_precomputed_datasets_conf_100 = dict(dataset_name='feverous',
                                             dataset_params=dict(
                                                 dataset_dir=DATASET_DIR,
                                                 dataset_file=[
                                                     'dev.combined.not_precomputed.p5.s5.t3_readable_test.jsonl',
                                                     'dev.combined.not_precomputed.p5.s5.t3_readable_train.jsonl',
                                                     'dev.combined.not_precomputed.p5.s20.t3_readable_test.jsonl',
                                                     'dev.combined.not_precomputed.p5.s20.t3_readable_train.jsonl',
                                                 ],
                                                 nrows=[100]),
                                             )

    politihop_xs_test = dict(dataset_name='politihop',
                             dataset_params=dict(
                                 dataset_dir=POLITIHOP_DS_DIR,
                                 dataset_file=[
                                     'politihop_test.tsv',
                                 ],
                                 nrows=[3]),
                             )
    LIARPlus_xs_test = dict(dataset_name='LIARPlus',
                            dataset_params=dict(
                                dataset_dir=LIARPLUS_DS_DIR,
                                dataset_file=[
                                    'test2.tsv',
                                ],
                                nrows=[3]),
                            )
    scifact_1k = du({'dataset_name': 'SciFact',
                     'dataset_params': {'dataset_dir': ospj(DATASET_DIR_V3, 'scifact/converted/'),
                                        'dataset_file': ['dev_k20_mink5.json', ], 'nrows': 1000}},
                    EXPLAINER_CNAMES_3L_V1)
    scifact_xs = copy.deepcopy(scifact_1k)
    scifact_xs['dataset_params'].update(nrows=3)
    scifact_s = copy.deepcopy(scifact_1k)
    scifact_s['dataset_params'].update(nrows=10)

    fm2_1k = du(dict(dataset_name='FM2',
                     dataset_params=dict(dataset_dir=ospj(DATASET_DIR_V3, 'FM2/converted/'),
                                         dataset_file=['dev.json'], nrows=1000),
                     ), EXPLAINER_CNAMES_2L_V1)
    fm2_xs = copy.deepcopy(fm2_1k)
    fm2_xs['dataset_params']['nrows'] = 3

    f2l_1k = du(dict(dataset_name='feverous2l',
                     dataset_params=dict(dataset_dir=ospj(DATASET_DIR_V3, 'feverous/converted'),
                                         dataset_file='test_2labels_from5-5-3-dev.json',
                                         nrows=1000),
                     ), EXPLAINER_CNAMES_2L_V1)
    f2l_1k_full = du(f2l_1k, dict(
        dataset_name='feverous2l_full',
        dataset_params=dict(
            dataset_file='test_2labels_from5-5-3-full.json',
            nrows=1000,
        )), EXPLAINER_CNAMES_2L_V1)

    f3l_1k = dict(dataset_name='feverous3l',
                  dataset_params=dict(dataset_dir=ospj(DATASET_DIR_V3, 'feverous/converted'),
                                      dataset_file='test_from5-5-3-dev.json',
                                      nrows=1000),
                  )

    f3l_1k_full = dict(dataset_name='feverous3l_full',
                       dataset_params=dict(dataset_dir=ospj(DATASET_DIR_V3, 'feverous/converted'),
                                           dataset_file='test_from5-5-3-full.json',
                                           nrows=1000),
                       )

    not_precomputed_datasets_conf_10 = dict(dataset_name='feverous',
                                            dataset_params=dict(
                                                dataset_dir=DATASET_DIR,
                                                dataset_file=[
                                                    'dev.combined.not_precomputed.p5.s5.t3_readable_test.jsonl',
                                                    'dev.combined.not_precomputed.p5.s5.t3_readable_train.jsonl',
                                                    'dev.combined.not_precomputed.p5.s20.t3_readable_test.jsonl',
                                                    'dev.combined.not_precomputed.p5.s20.t3_readable_train.jsonl',
                                                ],
                                                nrows=[10]),
                                            )

    AVERITEC = du(dict(dataset_name='AVERITEC',
                       dataset_params=dict(dataset_dir=ospj(DATASET_DIR_V3, 'AVeriTeC/converted/'),
                                           dataset_file=['dev_corrected.json', ], nrows=1000),
                       ), AV_CLASS_NAMES)

    fake_predictor = dict(model_name=['fake_predictor'])

    JF_feverous_model = dict(model_name=['default'], model_params=dict(
        model_path=['/homes/bussotti/feverous_work/feverousdata/models_fromjf270623or']), )

    baseline_feverous_model = dict(
        model_name=['default'],
        model_params=dict(
            model_path=[
                '/homes/bussotti/feverous_work/feverousdata/modeloriginalfeverousforandrea/feverous_verdict_predictor']),
    )

    llama3_1_v0 = dict(model_name=['LLAMA3_1'],
                       model_params=dict(base_model_name="meta-llama/Meta-Llama-3.1-8B", ),
                       )

    pepa_f2l = dict(model_name='GenFCExp',
                    model_params=dict(
                        model_path=ospj(MODELS_DIR_V2,
                                        'Isabelle/feverous/isabelle_fromfevtrain_2labels_from5-5-3-dev_doubt.pt'),
                        nlabels=2,
                    ))
    pepa_f2l_v2 = dict(model_name='GenFCExp_v2',
                       model_params=dict(
                           model_path=ospj(MODELS_DIR_V2,
                                           'Isabelle/feverous/fromfull5-5-3_2labels_6ep_lr3e-5.pt'),
                           nlabels=2,
                       ))

    pepa_f3l = dict(model_name='GenFCExp',
                    model_params=dict(
                        model_path=ospj(MODELS_DIR_V2,
                                        'Isabelle/feverous/isabelle_fromfevtrain_from5-5-3-dev_doubt.pt'),
                        nlabels=3,
                    ))
    pepa_f3l_v2 = dict(model_name='GenFCExp_v2',
                       model_params=dict(
                           model_path=ospj(MODELS_DIR_V2, 'Isabelle/feverous/fromfull5-5-3_3ep_lr3e-5.pt'),
                           nlabels=3,
                       ))
    pepa_scifact = dict(model_name='GenFCExp',
                        model_params=dict(
                            model_path=ospj(MODELS_DIR_V2, 'Isabelle/scifact/isabelle_k20_mink5_doubt.pt'),
                            nlabels=3, ),
                        )
    pepa_scifact_v2 = dict(model_name='GenFCExp_v2',
                           model_params=dict(
                               model_path=ospj(MODELS_DIR_V2, 'Isabelle/scifact/fromtrain_k20_mink5_3ep_lr3e-5.pt'),
                               nlabels=3, ),
                           )
    pepa_fm2 = dict(model_name=['GenFCExp'],
                    model_params=dict(model_path=ospj(MODELS_DIR_V2, 'Isabelle/FM2/isabelle_doubt.pt'), ),
                    )

    pepa_fm2_v2 = dict(model_name=['GenFCExp_v2'],
                       model_params=dict(model_path=ospj(MODELS_DIR_V2, 'Isabelle/FM2/fromtrain_6ep_lr3e-5.pt'),
                                         nlabels=2),
                       )

    pepa_av = {'model_name': 'GenFCExp',
               'model_params': {
                   'model_path': ospj(MODELS_DIR_V2, 'Isabelle/AVeriTeC/fromtrainnoise_equilibrated_2ep_lr3e-5.pt'),
                   'nlabels': 4}}

    pepa_script_2 = dict(model_params=dict(script_name='test_model_copy.sh'))
    pepa_script_3 = dict(model_params=dict(script_name='test_model_copy_2.sh'))
    pepa_script_4 = dict(model_params=dict(script_name='test_model_copy_3.sh'))
    pepa_script_5 = dict(model_params=dict(script_name='test_model_copy_4.sh'))
    pepa_script_6 = dict(model_params=dict(script_name='test_model_copy_5.sh'))

    roberta_base = dict(model_name=['Roberta'], model_params=dict(), )
    roberta_sci_fact = copy.deepcopy(roberta_base)
    roberta_sci_fact['model_params'].update(
        model_path=ospj(MODELS_DIR_V2, 'feverous/Scifact/checkpoint-549'),
        nb_label=3,
    )
    roberta_sci_fact_v2 = du(roberta_sci_fact, dict(
        model_name='Roberta_v2',
        model_params=dict(
            model_path=ospj(MODELS_DIR_V2,
                            'feverous/Scifact/from_train_k20_mink5_equilibrated_lr1e-05_3ep'),
            nb_label=3)))

    roberta_fm2_v2 = du(roberta_base, dict(
        model_name='Roberta_v2',
        model_params=dict(
            model_path=ospj(MODELS_DIR_V2, 'feverous/FM2/fromtrain_3ep_lr1e-07'),
            nb_label=2)))

    roberta_fm2_v1 = du(roberta_base, dict(
        model_name='Roberta',
        model_params=dict(
            model_path=ospj(MODELS_DIR_V2, 'feverous/FM2/RM_fromtrain_1ep_lr1e-05/checkpoint-9991'),
            nb_label=2)))

    roberta_f2l_noise = du(roberta_base, dict(model_params=dict(
        model_path=ospj(MODELS_DIR_V2, 'feverous/feverous/fromfevtrain_2labels_from5-5-3-full_1ep_lr1e-07'),
        nb_label=2,
    )))

    roberta_f2l_no_noise = du(roberta_base, dict(
        model_name='Roberta_v2_no_noise',
        model_params=dict(
            model_path=ospj(MODELS_DIR_V2, 'feverous/feverous/fromfevtrain_2labels_nonoisefull_1ep_lr1e-07'),
            nb_label=2,
        )))

    roberta_f3l_noise = copy.deepcopy(roberta_base)
    roberta_f3l_noise['model_params'].update(
        model_path=ospj(MODELS_DIR_V2, 'feverous/feverous/fromfevtrain_from5-5-3-full_1ep_lr1e-05'),
        nb_label=3,
    )

    roberta_f3l_no_noise = du(roberta_base, dict(
        model_name='Roberta_v2_no_noise',
        model_params=dict(
            model_path=ospj(MODELS_DIR_V2, 'feverous/feverous/fromfevtrain_nonoisefull_1ep_lr1e-05'),
            nb_label=3,
        )))

    roberta_averitec_v1 = du(roberta_base, dict(
        model_name='Roberta',
        model_params=dict(
            model_path=ospj(MODELS_DIR_V2, 'feverous/AVeriTeC/from_train_lr1e-05_1ep'),
            nb_label=4, batch_size=128)))

    llama8B_v1 = {'model_name': 'LLAMA31_8B',
                  'model_params': {
                      # 'cache_pred_file': ospj(BASE_V1, 'cache_llama8B.json'),
                      'prompt_type': 'noEvidence',
                      'modelToUse': 'llama3.1',
                      'nb_ex_fs': 0,
                      'input_for_fs': ospj(DATASET_DIR_V3, 'feverous/converted/test_2labels_from5-5-3-dev.json')}
                  }

    llama8B_f2l = du(llama8B_v1, {'model_params': {
        'cache_pred_file': ospj(LLAMA_CACHE, 'cache_llama8B_f2l.json'),
        'labels': 'REFUTES.SUPPORTS'}})

    llama70B_v1 = {'model_name': 'LLAMA31_70B',
                   'model_params': {
                       # 'cache_pred_file': ospj(BASE_V1, 'cache_llama8B.json'),
                       'prompt_type': 'noEvidence',
                       'modelToUse': 'llama3.1:70b',
                       'nb_ex_fs': 0,
                       'input_for_fs': ospj(DATASET_DIR_V3, 'feverous/converted/test_2labels_from5-5-3-dev.json')}
                   }
    llama70B_f2l = du(llama70B_v1, {'model_params': {
        'cache_pred_file': ospj(LLAMA_CACHE, 'cache_llama70B_f2l.json'),
        'labels': 'REFUTES.SUPPORTS',
        'default_label': 'REFUTES'}})
    llama70B_f3l = du(llama70B_v1, {'model_params': {
        'cache_pred_file': ospj(LLAMA_CACHE, 'cache_llama70B_f3l.json'),
        'labels': 'REFUTES.NOT_ENOUGH_INFO.SUPPORTS',
        'default_label': 'NOT_ENOUGH_INFO'}})

    llama70B_sf = du(llama70B_v1, {'model_params': {
        'cache_pred_file': ospj(LLAMA_CACHE, 'cache_llama70B_sf.json'),
        'labels': 'REFUTES.NOT_ENOUGH_INFO.SUPPORTS',
        'default_label': 'NOT_ENOUGH_INFO'}})

    llama70B_fm2 = du(llama70B_v1, {'model_params': {
        'cache_pred_file': ospj(LLAMA_CACHE, 'cache_llama70B_fm2.json'),
        'labels': 'REFUTES.SUPPORTS',
        'default_label': 'REFUTES'}})

    llama70B_av = du(llama70B_v1, {'model_params': {
        'cache_pred_file': ospj(LLAMA_CACHE, 'cache_llama70B_av.json'),
        'labels': '.'.join(AV_CLASS_NAMES['explainer_params']['class_names']),
        'default_label': 'Not_Enough_Evidence'}})

    BS_64 = dict(model_params=dict(batch_size=64))

    plain_pred_exp = {'explainer_name': 'plain_pred'}
    lime_only_ev_v1 = dict(explainer_name=['lime'],
                           explainer_params=dict(perturbation_mode=['only_evidence'], num_samples=[500],
                                                 class_names=CLASS_NAMES_V1),
                           )
    ns_100 = {'explainer_params': {'num_samples': [100]}}
    lime_only_ev_v0 = du(lime_only_ev_v1, dict(explainer_params=dict(class_names=CLASS_NAMES_V0)))
    lime_only_ev_250 = copy.deepcopy(lime_only_ev_v0)
    lime_only_ev_250['explainer_params'].update(num_samples=[250])

    lime_only_ev_50 = copy.deepcopy(lime_only_ev_v0)
    lime_only_ev_50['explainer_params'].update(num_samples=[50])

    lime_only_ev_stability_s = copy.deepcopy(lime_only_ev_v0)
    lime_only_ev_stability_s['explainer_params'].update(num_samples=[8, 16, 32])

    lime_only_ev_stability = copy.deepcopy(lime_only_ev_v0)
    lime_only_ev_stability['explainer_params'].update(num_samples=[16, 32, 64, 128, 256, 512])

    shap_only_ev_v1 = dict(explainer_name=['shap'],
                           explainer_params=dict(perturbation_mode=['only_evidence'], mode=['KernelExplainer'],
                                                 num_samples=[500],
                                                 class_names=CLASS_NAMES_V1),
                           )
    shap_only_ev_v0 = du(shap_only_ev_v1, dict(explainer_params=dict(class_names=CLASS_NAMES_V0)))
    shap_only_ev_250 = copy.deepcopy(shap_only_ev_v0)
    shap_only_ev_250['explainer_params'].update(num_samples=[250])

    shap_only_ev_50 = copy.deepcopy(shap_only_ev_v0)
    shap_only_ev_50['explainer_params'].update(num_samples=[50])

    shap_only_ev_stability = copy.deepcopy(shap_only_ev_v0)
    shap_only_ev_stability['explainer_params'].update(num_samples=[16, 32, 64, 128, 256, 512])

    n_perturb_time = [int(x) for x in (2 ** np.arange(5, 13 + 1))]

    lime_only_ev_time = copy.deepcopy(lime_only_ev_v0)
    lime_only_ev_time['explainer_params'].update(num_samples=n_perturb_time)

    lime_only_ev_time_v2_s = copy.deepcopy(lime_only_ev_v0)
    lime_only_ev_time_v2_s['explainer_params'].update(num_samples=n_perturb_time[:3])

    shap_only_ev_time = copy.deepcopy(shap_only_ev_v0)
    shap_only_ev_time['explainer_params'].update(num_samples=n_perturb_time)

    shap_only_ev_time_v2_s = copy.deepcopy(shap_only_ev_v0)
    shap_only_ev_time_v2_s['explainer_params'].update(num_samples=n_perturb_time[:3])

    claim_only_explainer = dict(explainer_name=['claim_only_pred'], explainer_params=dict(), )


REQUIRED_FIELDS = ['experiment_id', 'dataset_name', 'model_name', 'explainer_name',
                   'dataset_params.dataset_file'  # will be used to identify the dataset
                   ]

experiment_definitions_list = [
    dict(experiment_id='sk_f_jf_1.0', ) | E.BASE_CONFIG |
    E.JF_feverous_model | E.lime_only_ev_v0 | E.feverous_datasets_conf,

    dict(experiment_id='sk_f_jf_1.1', ) | E.BASE_CONFIG |
    E.JF_feverous_model | E.shap_only_ev_v0 | dict(dataset_name='feverous',
                                                   dataset_params=dict(
                                                       dataset_dir=E.BASE_V1,
                                                       dataset_file=[
                                                           'ex_AB_00.jsonl',
                                                       ],
                                                       nrows=[1000]),
                                                   ),
    dict(experiment_id='sk_f_jf_1.1b', ) | E.BASE_CONFIG |
    E.JF_feverous_model | E.shap_only_ev_v0 | dict(dataset_name='feverous',
                                                   dataset_params=dict(
                                                       dataset_dir=E.BASE_V1,
                                                       dataset_file=[
                                                           'original_TO_01_formatted.jsonl',
                                                           'feverous_dev_ST_01.jsonl',
                                                           'feverous_dev_SO_01.jsonl',
                                                       ],
                                                       nrows=[1000]),
                                                   ),
    dict(experiment_id='sk_f_jf_1.1n', ) | E.BASE_CONFIG |
    E.JF_feverous_model | E.shap_only_ev_v0 | dict(dataset_name='feverous',
                                                   dataset_params=dict(
                                                       dataset_dir=E.BASE_V1,
                                                       dataset_file=[
                                                           'feverous_train_challenges_withnoise.jsonl',
                                                       ],
                                                       nrows=[1000]),
                                                   ),

    dict(experiment_id='f_bs_1.0', ) | E.BASE_CONFIG |
    E.baseline_feverous_model | E.lime_only_ev_v0 | E.feverous_datasets_conf,

    dict(experiment_id='f_bs_1.1', ) | E.BASE_CONFIG |
    E.baseline_feverous_model | E.shap_only_ev_v0 | dict(dataset_name='feverous',
                                                         dataset_params=dict(
                                                             dataset_dir=E.BASE_V1,
                                                             dataset_file=[
                                                                 'ex_AB_00.jsonl',
                                                                 'feverous_train_challenges_withnoise.jsonl',
                                                             ],
                                                             nrows=[1000]),
                                                         ),

    dict(experiment_id='f_bs_1.1b', ) | E.BASE_CONFIG |
    E.baseline_feverous_model | E.shap_only_ev_v0 | dict(dataset_name='feverous',
                                                         dataset_params=dict(
                                                             dataset_dir=E.BASE_V1,
                                                             dataset_file=[
                                                                 'feverous_dev_SO_01.jsonl',
                                                                 'original_TO_01_formatted.jsonl',
                                                             ],
                                                             nrows=[1000]),
                                                         ),
    dict(experiment_id='f_bs_1.1c', ) | E.BASE_CONFIG |
    E.baseline_feverous_model | E.shap_only_ev_v0 | dict(dataset_name='feverous',
                                                         dataset_params=dict(
                                                             dataset_dir=E.BASE_V1,
                                                             dataset_file=["feverous_dev_ST_01.jsonl", ],
                                                             nrows=[1000]),
                                                         ),

    dict(experiment_id='oc_1.0', ) | E.BASE_CONFIG |
    E.JF_feverous_model | E.claim_only_explainer | E.feverous_datasets_conf,

    dict(experiment_id='oc_1.1', ) | E.BASE_CONFIG |
    E.baseline_feverous_model | E.claim_only_explainer | E.feverous_datasets_conf,

    # retrived datasets not_precomputed
    dict(experiment_id='fbs_np_1.0', ) | E.BASE_CONFIG |
    E.baseline_feverous_model | E.lime_only_ev_v0 | E.not_precomputed_datasets_conf,

    dict(experiment_id='fbs_np_2.0', ) | E.BASE_CONFIG |
    E.baseline_feverous_model | E.shap_only_ev_v0 | E.not_precomputed_datasets_conf,

    # LLAMA3.1 on noisy datasets
    dict(experiment_id='lla_np_1.0', ) | E.BASE_CONFIG |
    E.llama3_1_v0 | E.lime_only_ev_250 | E.not_precomputed_datasets_conf_100,

    dict(experiment_id='lla_np_2.0', ) | E.BASE_CONFIG |
    E.llama3_1_v0 | E.shap_only_ev_250 | E.not_precomputed_datasets_conf_100,

    # LLAMA3.1 on normal feverous datasets
    dict(experiment_id='lla_fv_1.0', ) | E.BASE_CONFIG |
    E.llama3_1_v0 | E.lime_only_ev_250 | E.feverous_datasets_conf_100,

    dict(experiment_id='lla_fv_1.1', ) | E.BASE_CONFIG |
    E.llama3_1_v0 | E.shap_only_ev_250 | E.feverous_datasets_conf_100,

    dict(experiment_id='lla_fv_1.2', ) | E.BASE_CONFIG |
    E.llama3_1_v0 | E.claim_only_explainer | E.feverous_datasets_conf_100,
    # end LLAMA3.1 on normal datasets

    dict(experiment_id='lla_np_1.test', ) | E.BASE_CONFIG |
    E.llama3_1_v0 | E.shap_only_ev_250 | dict(dataset_name='feverous',
                                              dataset_params=dict(
                                                  dataset_dir=E.DATASET_DIR,
                                                  dataset_file=[
                                                      'dev.combined.not_precomputed.p5.s5.t3_readable_test.jsonl',
                                                      'dev.combined.not_precomputed.p5.s5.t3_readable_train.jsonl',
                                                      'dev.combined.not_precomputed.p5.s20.t3_readable_test.jsonl',
                                                      'dev.combined.not_precomputed.p5.s20.t3_readable_train.jsonl',
                                                  ],
                                                  nrows=[10], skiprows=[2]),
                                              ) | dict(random_seed=[3]),

    dict(experiment_id='oc_fbs_np_1.0', ) | E.BASE_CONFIG |
    E.baseline_feverous_model | E.claim_only_explainer | E.not_precomputed_datasets_conf,

    # time scalability
    dict(experiment_id='fbs_time_1.0', ) | E.BASE_CONFIG |
    E.baseline_feverous_model | E.lime_only_ev_time | E.feverous_ds_100,

    dict(experiment_id='fbs_time_2.0', ) | E.BASE_CONFIG |
    E.baseline_feverous_model | E.shap_only_ev_time | E.feverous_ds_100,

    dict(experiment_id='fbs_time_1.1', ) | E.BASE_CONFIG |
    E.baseline_feverous_model | E.lime_only_ev_time_v2_s | E.feverous_ds_100,

    dict(experiment_id='fbs_time_2.1', ) | E.BASE_CONFIG |
    E.baseline_feverous_model | E.shap_only_ev_time_v2_s | E.feverous_ds_100,

    dict(experiment_id='fbs_time_2.1test', ) | E.BASE_CONFIG |
    E.baseline_feverous_model | E.shap_only_ev_time_v2_s | E.feverous_ds_100,
    # after having the results of the time experiment
    # define the best number of samples experiment with less combinations of num_samples.

    dict(experiment_id='sms_p_1.0', ) | E.BASE_CONFIG |
    E.fake_predictor |
    E.lime_only_ev_50
    # dict(explainer_name=['lime'],
    #      explainer_params=dict(perturbation_mode=['only_evidence'], num_samples=[50], separator=r'<|reserved_special_token_15|>', ),
    #      )
    | E.feverous_ds_xs
    # C.politihop_10_test
    ,

    # dict(experiment_id='test_3.0', ) | C.BASE_CONFIG | C.fake_predictor | C.lime_only_ev_50 | C.feverous_ds_xs,
    # dict(experiment_id='test_3.1', ) | C.BASE_CONFIG | C.fake_predictor | C.shap_only_ev_50 | C.feverous_ds_xs,
    dict(experiment_id='test_2.0', ) | E.BASE_CONFIG | E.fake_predictor | E.lime_only_ev_50 | E.feverous_ds_xs,
    dict(experiment_id='test_2.1', ) | E.BASE_CONFIG | E.fake_predictor | E.shap_only_ev_50 | E.feverous_ds_xs,
    dict(experiment_id='test_1.0', ) | E.BASE_CONFIG | E.fake_predictor | E.lime_only_ev_50 | E.politihop_xs_test,
    dict(experiment_id='test_1.1', ) | E.BASE_CONFIG | E.fake_predictor | E.shap_only_ev_50 | E.politihop_xs_test,
    dict(experiment_id='test_3.0', ) | E.BASE_CONFIG | E.fake_predictor | E.lime_only_ev_50 | E.LIARPlus_xs_test,

    # stability test
    dict(experiment_id='st_1.0', ) | E.BASE_CONFIG | E.RANDOM_SEEDS_v1 |
    E.baseline_feverous_model | E.lime_only_ev_stability_s | E.feverous_ds_xs,

    dict(experiment_id='st_1.1', ) | E.BASE_CONFIG | E.RANDOM_SEEDS_v1 |
    E.baseline_feverous_model | E.lime_only_ev_stability | E.feverous_ds_100,

    dict(experiment_id='st_1.2', ) | E.BASE_CONFIG | E.RANDOM_SEEDS_v1 |
    E.baseline_feverous_model | E.shap_only_ev_stability | E.feverous_ds_100,

    # NEW DATASETS
    # FEVEROUS 2 label
    du(dict(experiment_id='fv_f2l_1.0', ), E.BASE_CONFIG,
       E.roberta_f2l_noise, E.lime_only_ev_v1, E.f2l_1k, E.EXPLAINER_CNAMES_2L_V1),

    du(dict(experiment_id='fv_f2l_2.0', ), E.BASE_CONFIG,
       E.roberta_f2l_noise, E.shap_only_ev_v1, E.f2l_1k, E.EXPLAINER_CNAMES_2L_V1),

    du(dict(experiment_id='fv_f2l_3.0', ), E.BASE_CONFIG,
       E.roberta_f2l_noise, E.claim_only_explainer, E.f2l_1k, E.EXPLAINER_CNAMES_2L_V1),

    # FEVEROUS 3 label
    dict(experiment_id='fv_f3l_1.0', ) | E.BASE_CONFIG |
    E.roberta_f3l_noise | E.lime_only_ev_v1 | E.f3l_1k,

    dict(experiment_id='fv_f3l_2.0', ) | E.BASE_CONFIG |
    E.roberta_f3l_noise | E.shap_only_ev_v1 | E.f3l_1k,

    # FEVEROUS 2 label FULL
    du(dict(experiment_id='fv_f2lF_1.0', ), E.BASE_CONFIG,
       E.roberta_f2l_noise, E.lime_only_ev_v1, E.f2l_1k_full, E.EXPLAINER_CNAMES_2L_V1, E.BS_64),

    du(dict(experiment_id='fv_f2lF_2.0', ), E.BASE_CONFIG,
       E.roberta_f2l_noise, E.shap_only_ev_v1, E.f2l_1k_full, E.EXPLAINER_CNAMES_2L_V1, E.BS_64),

    # FEVEROUS 3 label FULL
    du(dict(experiment_id='fv_f3lF_1.0', ), E.BASE_CONFIG,
       E.roberta_f3l_noise, E.lime_only_ev_v1, E.f3l_1k_full, E.BS_64),

    du(dict(experiment_id='fv_f3lF_2.0', ), E.BASE_CONFIG,
       E.roberta_f3l_noise, E.shap_only_ev_v1, E.f3l_1k_full, E.BS_64),

    # scifact
    dict(experiment_id='fv_sf_1.0', ) | E.BASE_CONFIG |
    E.roberta_sci_fact | E.lime_only_ev_v1 | E.scifact_1k,

    dict(experiment_id='fv_sf_2.0', ) | E.BASE_CONFIG |
    E.roberta_sci_fact | E.shap_only_ev_v1 | E.scifact_1k,

    dict(experiment_id='fv_sf_1.0test', ) | E.BASE_CONFIG |
    E.roberta_sci_fact_v2 | E.plain_pred_exp | E.scifact_1k,

    # FM2
    du(dict(experiment_id='fv_fm_1.0', ), E.BASE_CONFIG,
       E.roberta_fm2_v1, E.lime_only_ev_v1, E.fm2_1k, E.EXPLAINER_CNAMES_2L_V1),
    du(dict(experiment_id='fv_fm_2.0', ), E.BASE_CONFIG,
       E.roberta_fm2_v1, E.shap_only_ev_v1, E.fm2_1k, E.EXPLAINER_CNAMES_2L_V1),

    du(dict(experiment_id='fv_fm_1.0test', ), E.BASE_CONFIG,
       E.roberta_fm2_v1, E.lime_only_ev_50, E.fm2_xs, E.EXPLAINER_CNAMES_2L_V1),

    # AVERITEC
    du(dict(experiment_id='fv_av_1.0', ), E.BASE_CONFIG,
       E.roberta_averitec_v1, E.lime_only_ev_v1, E.AVERITEC, E.AV_CLASS_NAMES),
    du(dict(experiment_id='fv_av_2.0', ), E.BASE_CONFIG,
       E.roberta_averitec_v1, E.shap_only_ev_v1, E.AVERITEC, E.AV_CLASS_NAMES),

    # round 2
    # FEVEROUS 2l
    du(dict(experiment_id='r2_fv_f2l_1.0', ), E.BASE_CONFIG,
       E.roberta_f2l_no_noise, E.lime_only_ev_v1, E.f2l_1k, E.EXPLAINER_CNAMES_2L_V1),

    du(dict(experiment_id='r2_fv_f2l_2.0', ), E.BASE_CONFIG,
       E.roberta_f2l_no_noise, E.shap_only_ev_v1, E.f2l_1k, E.EXPLAINER_CNAMES_2L_V1),

    # FEVEROUS 3l
    du(dict(experiment_id='r2_fv_f3l_1.0', ), E.BASE_CONFIG,
       E.roberta_f3l_no_noise, E.lime_only_ev_v1, E.f3l_1k),

    du(dict(experiment_id='r2_fv_f3l_2.0', ), E.BASE_CONFIG,
       E.roberta_f3l_no_noise, E.shap_only_ev_v1, E.f3l_1k),

    # FEVEROUS 2l FULL
    du(dict(experiment_id='r2_fv_f2lF_1.0', ), E.BASE_CONFIG, E.BS_64,
       E.roberta_f2l_no_noise, E.lime_only_ev_v1, E.f2l_1k_full, E.EXPLAINER_CNAMES_2L_V1),

    du(dict(experiment_id='r2_fv_f2lF_2.0', ), E.BASE_CONFIG, E.BS_64,
       E.roberta_f2l_no_noise, E.shap_only_ev_v1, E.f2l_1k_full, E.EXPLAINER_CNAMES_2L_V1),

    # FEVEROUS 3l
    du(dict(experiment_id='r2_fv_f3lF_1.0', ), E.BASE_CONFIG, E.BS_64,
       E.roberta_f3l_no_noise, E.lime_only_ev_v1, E.f3l_1k_full),

    du(dict(experiment_id='r2_fv_f3lF_2.0', ), E.BASE_CONFIG, E.BS_64,
       E.roberta_f3l_no_noise, E.shap_only_ev_v1, E.f3l_1k_full),

    # Scifact
    du(dict(experiment_id='r2_fv_sf_1.0', ), E.BASE_CONFIG,
       E.roberta_sci_fact_v2, E.lime_only_ev_v1, E.scifact_1k, E.BS_64),

    du(dict(experiment_id='r2_fv_sf_2.0', ), E.BASE_CONFIG,
       E.roberta_sci_fact_v2, E.shap_only_ev_v1, E.scifact_1k, E.BS_64),

    # FM2
    du(dict(experiment_id='r2_fv_fm_1.0', ), E.BASE_CONFIG,
       E.roberta_fm2_v2, E.lime_only_ev_v1, E.fm2_1k, E.EXPLAINER_CNAMES_2L_V1),
    du(dict(experiment_id='r2_fv_fm_2.0', ), E.BASE_CONFIG,
       E.roberta_fm2_v2, E.shap_only_ev_v1, E.fm2_1k, E.EXPLAINER_CNAMES_2L_V1),

    # Isabelle/Pepa Generating Fact-Checking Explanations
    # FEVEROUS 2 label
    du(dict(experiment_id='gfce_f2l_1.0', ), E.BASE_CONFIG,
       E.pepa_f2l, E.lime_only_ev_50, E.f2l_1k, E.EXPLAINER_CNAMES_2L_V1),
    du(dict(experiment_id='gfce_f2l_2.0', ), E.BASE_CONFIG,
       E.pepa_f2l, E.shap_only_ev_v1, E.f2l_1k, E.EXPLAINER_CNAMES_2L_V1),

    du(dict(experiment_id='gfce_f2l_1.1', ), E.BASE_CONFIG,
       E.pepa_f2l_v2, E.lime_only_ev_v1, E.f2l_1k, E.EXPLAINER_CNAMES_2L_V1, E.pepa_script_3),
    du(dict(experiment_id='gfce_f2l_2.1', ), E.BASE_CONFIG,
       E.pepa_f2l_v2, E.shap_only_ev_v1, E.f2l_1k, E.EXPLAINER_CNAMES_2L_V1, E.pepa_script_4),

    du(dict(experiment_id='gfce_f2l_1.1F', ), E.BASE_CONFIG,
       E.pepa_f2l_v2, E.lime_only_ev_v1, E.f2l_1k_full, E.EXPLAINER_CNAMES_2L_V1, ),
    du(dict(experiment_id='gfce_f2l_2.1F', ), E.BASE_CONFIG,
       E.pepa_f2l_v2, E.shap_only_ev_v1, E.f2l_1k_full, E.EXPLAINER_CNAMES_2L_V1, E.pepa_script_2),

    # FEVEROUS 3 label
    du(dict(experiment_id='gfce_f3l_1.0', ), E.BASE_CONFIG,  # OK 4:27h
       E.pepa_f3l, E.lime_only_ev_50, E.f3l_1k_full),
    du(dict(experiment_id='gfce_f3l_2.0', ), E.BASE_CONFIG,
       E.pepa_f3l, E.shap_only_ev_v1, E.f3l_1k_full),

    du(dict(experiment_id='gfce_f3l_1.1', ), E.BASE_CONFIG,
       E.pepa_f3l_v2, E.lime_only_ev_50, E.f3l_1k_full, E.pepa_script_2),
    du(dict(experiment_id='gfce_f3l_2.1', ), E.BASE_CONFIG,
       E.pepa_f3l_v2, E.shap_only_ev_v1, E.f3l_1k_full, E.pepa_script_3),

    du(dict(experiment_id='gfce_f3l_1.1F', ), E.BASE_CONFIG,
       E.pepa_f3l_v2, E.lime_only_ev_v1, E.f3l_1k_full, E.pepa_script_5),

    # Scifact
    du(dict(experiment_id='gfce_sf_1.0', ), E.BASE_CONFIG,  # OK 1:10h
       E.pepa_scifact, E.lime_only_ev_50, E.scifact_1k, E.pepa_script_2),
    du(dict(experiment_id='gfce_sf_2.0', ), E.BASE_CONFIG,  # Errors. Skip. Bad model version
       E.pepa_scifact, E.shap_only_ev_v1, E.scifact_1k, E.pepa_script_2),

    du(dict(experiment_id='gfce_sf_1.1', ), E.BASE_CONFIG,
       E.pepa_scifact_v2, E.lime_only_ev_50, E.scifact_1k, E.pepa_script_4),
    du(dict(experiment_id='gfce_sf_2.1', ), E.BASE_CONFIG,
       E.pepa_scifact_v2, E.shap_only_ev_v1, E.scifact_1k, E.pepa_script_4),
    du(dict(experiment_id='gfce_sf_1.1F', ), E.BASE_CONFIG,
       E.pepa_scifact_v2, E.lime_only_ev_v1, E.scifact_1k),

    # FM2
    du(dict(experiment_id='gfce_fm2_1.0', ), E.BASE_CONFIG,  # OK 4:29h
       E.pepa_fm2, E.lime_only_ev_50, E.fm2_1k, E.EXPLAINER_CNAMES_2L_V1, E.pepa_script_3),
    du(dict(experiment_id='gfce_fm2_2.0', ), E.BASE_CONFIG,  # Errors. Skip. Bad model version
       E.pepa_fm2, E.shap_only_ev_v1, E.fm2_1k, E.EXPLAINER_CNAMES_2L_V1, E.pepa_script_3),

    du(dict(experiment_id='gfce_fm2_1.1', ), E.BASE_CONFIG,
       E.pepa_fm2_v2, E.lime_only_ev_v1, E.fm2_1k, E.EXPLAINER_CNAMES_2L_V1, ),
    du(dict(experiment_id='gfce_fm2_2.1', ), E.BASE_CONFIG,
       E.pepa_fm2_v2, E.shap_only_ev_v1, E.fm2_1k, E.EXPLAINER_CNAMES_2L_V1, E.pepa_script_2),
    du(dict(experiment_id='gfce_fm2_1.2', ), E.BASE_CONFIG,
       E.pepa_fm2_v2, E.lime_only_ev_v1, E.fm2_1k, {'dataset_params': {'nrows': None}}),
    du(dict(experiment_id='gfce_fm2_2.2', ), E.BASE_CONFIG, E.pepa_script_2,
       E.pepa_fm2_v2, E.shap_only_ev_v1, E.fm2_1k, {'dataset_params': {'nrows': None}}),

    du(dict(experiment_id='gfce_sf_1.1test', ), E.BASE_CONFIG,
       E.pepa_scifact, E.lime_only_ev_50, E.scifact_xs),

    # AVERITEC
    du(dict(experiment_id='gfce_av_1.0', ), E.BASE_CONFIG,
       E.pepa_av, E.lime_only_ev_50, E.AVERITEC, E.AV_CLASS_NAMES, E.pepa_script_3),
    du(dict(experiment_id='gfce_av_2.0', ), E.BASE_CONFIG,
       E.pepa_av, E.shap_only_ev_v1, E.AVERITEC, E.AV_CLASS_NAMES, E.pepa_script_2),
    du(dict(experiment_id='gfce_av_1.0F', ), E.BASE_CONFIG,
       E.pepa_av, E.lime_only_ev_v1, E.AVERITEC, E.AV_CLASS_NAMES, E.pepa_script_4),

    # LLAMA 8B
    # FEVEROUS 2 label
    du(dict(experiment_id='llama8b_f2l_1.0', ), E.BASE_CONFIG,
       E.llama8B_f2l, E.lime_only_ev_v1, E.f2l_1k_full, E.EXPLAINER_CNAMES_2L_V1),
    du(dict(experiment_id='llama8b_f2l_2.0', ), E.BASE_CONFIG,
       E.llama8B_f2l, E.shap_only_ev_v1, E.f2l_1k_full, E.EXPLAINER_CNAMES_2L_V1),

    # LLAMA 70B
    # FEVEROUS 2 label
    du(dict(experiment_id='llama70b_f2l_1.0', ), E.BASE_CONFIG,
       E.llama70B_f2l, E.lime_only_ev_v1, E.f2l_1k_full, E.ns_100),
    du(dict(experiment_id='llama70b_f2l_1.1', ), E.BASE_CONFIG,
       E.llama70B_f2l, E.lime_only_ev_v1, E.f2l_1k_full, E.ns_100),
    du(dict(experiment_id='llama70b_f2l_2.1', ), E.BASE_CONFIG,
       E.llama70B_f2l, E.shap_only_ev_v1, E.f2l_1k_full, E.ns_100),

    # FEVEROUS 3 label
    du(dict(experiment_id='llama70b_f3l_1.0', ), E.BASE_CONFIG,
       E.llama70B_f3l, E.lime_only_ev_v1, E.f3l_1k_full, E.ns_100),
    du(dict(experiment_id='llama70b_f3l_2.0', ), E.BASE_CONFIG,
       E.llama70B_f3l, E.shap_only_ev_v1, E.f3l_1k_full, E.ns_100),

    # SciFact
    du(dict(experiment_id='llama70b_sf_1.0', ), E.BASE_CONFIG,
       E.llama70B_sf, E.lime_only_ev_v1, E.scifact_1k, E.ns_100),
    du(dict(experiment_id='llama70b_sf_2.0', ), E.BASE_CONFIG,
       E.llama70B_sf, E.shap_only_ev_v1, E.scifact_1k, E.ns_100),

    # FM2
    du(dict(experiment_id='llama70b_fm2_1.0', ), E.BASE_CONFIG,
       E.llama70B_fm2, E.lime_only_ev_v1, E.fm2_1k, E.ns_100),
    du(dict(experiment_id='llama70b_fm2_2.0', ), E.BASE_CONFIG,
       E.llama70B_fm2, E.shap_only_ev_v1, E.fm2_1k, E.ns_100),

    # AVERITEC
    du(dict(experiment_id='llama70b_av_1.0', ), E.BASE_CONFIG,
       E.llama70B_av, E.lime_only_ev_v1, E.AVERITEC, E.ns_100),
    du(dict(experiment_id='llama70b_av_2.0', ), E.BASE_CONFIG,
       E.llama70B_av, E.shap_only_ev_v1, E.AVERITEC, E.ns_100),

]

check_experiments(experiment_definitions_list, REQUIRED_FIELDS)
