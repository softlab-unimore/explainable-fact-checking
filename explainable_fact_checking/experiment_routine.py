import copy
import os

import pydantic.utils

from explainable_fact_checking.experiment_definitions import E

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA_DEVICE_ORDER:", os.environ.get("CUDA_DEVICE_ORDER"))

import torch

print("CUDA is available:", torch.cuda.is_available())

import gc
import itertools
import json
import pickle
import shutil
import sys
from datetime import datetime
from tqdm import tqdm
import explainable_fact_checking as xfc
from explainable_fact_checking import models


class ExperimentRunner:

    @staticmethod
    def product_dict(**kwargs):
        keys = kwargs.keys()
        # check that all values are non-empty lists
        for k, v in kwargs.items():
            if not isinstance(v, list):
                raise TypeError(f"Value {v} is not a valid collection for key {k}")
            if len(v) == 0:
                raise ValueError(f"Value {v} cannot be empty for key {k}")
        return [dict(zip(keys, instance)) for instance in itertools.product(*kwargs.values())]

    @staticmethod
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
            assert len(experiment_definitions_dict) == len(
                experiment_definitions_list), 'Experiment ids are not unique.'
        else:
            raise ValueError('Invalid experiment_definitions type. It must be a list or a dictionary.')

        exp_dict = experiment_definitions_dict.get(experiment_id, None)
        if exp_dict is None:
            raise ValueError(f'Experiment id {experiment_id} not found in the configuration file.')

        return exp_dict

    def launch_experiment_by_id(self, experiment_id: str, config_file_path=None, additional_params=None):
        exp_dict = self.get_config_by_id(experiment_id, config_file_path)
        if additional_params:
            exp_dict = pydantic.utils.deep_update(exp_dict, additional_params)
        self.launch_experiment_by_config(exp_dict)

    def launch_experiment_by_config(self, exp_dict: dict):
        if exp_dict is None:
            raise ValueError(f"{exp_dict} is not a valid experiment id")
        experiment_id: str = exp_dict.get('experiment_id')
        if not isinstance(experiment_id, str):
            raise ValueError('You must specify an experiment id')
        for attr in xfc.experiment_definitions.REQUIRED_FIELDS:
            tdict = exp_dict
            for subattr in attr.split('.'):
                if subattr not in tdict.keys():
                    raise ValueError(f'You must specify some value for {attr} parameter. It\'s empty.')
                tdict = tdict[subattr]

        results_dir: str = exp_dict.get('results_dir')
        if not isinstance(results_dir, str):
            raise ValueError('You must specify a valid path for the results directory')
        results_dir = os.path.join(results_dir, experiment_id)

        # generate all the possible combinations of the parameters
        # create a folder for each combination
        # write a json file with the parameters in the folder
        params_dict_to_iterate = {}
        for key, value in exp_dict.items():
            # if value is not a collection, make it a collection
            if isinstance(value, dict):
                # check that the values are all lists
                for k, v in value.items():
                    # # if str convert to list
                    # if isinstance(v, str):
                    #     value[k] = [v]
                    if not isinstance(v, list):
                        value[k] = [v]

                        # raise TypeError(f"Value {v} is not a valid collection")
                value = self.product_dict(**value)
            elif not isinstance(value, list):
                value = [value]
            else:
                try:
                    iterator = iter(value)
                except TypeError:
                    raise TypeError(f"Value {value} is not a valid collection")
            params_dict_to_iterate[key] = value
        os.makedirs(results_dir, exist_ok=True)
        # logger = xfc.xfc_utils.init_logger(save_dir=results_dir)

        logger = xfc.xfc_utils.LoggerSingleton(save_dir=results_dir, reset=True)
        logger.info(f"Starting experiment {experiment_id}")
        start_time = datetime.now()
        last_checkpoint_time = start_time
        to_iter = list(enumerate(self.product_dict(**params_dict_to_iterate)))
        for i, kwargs in to_iter:
            gc.collect()
            logger.info(f"Iteration: {i} with parameters: ")
            print(json.dumps(kwargs, indent=4, sort_keys=True))
            kwargs = copy.deepcopy(kwargs)
            kwargs['results_dir'] = os.path.join(results_dir, str(i))

            try:
                self.launch_single_experiment(**kwargs, logger=logger)
            except Exception as e:
                logger.error(f"Error in iteration {i}.")
                logger.error(e)
                gettrace = getattr(sys, 'gettrace', None)
                if gettrace is None:
                    pass
                elif gettrace():
                    raise e

            actual_time = datetime.now()
            tot_avg = (actual_time - start_time) / (i + 1)
            tot_remaining = tot_avg * (len(to_iter) - i - 1)

            checkpoint_avg = (actual_time - last_checkpoint_time) / (i + 1)
            remaining_checkpoint = checkpoint_avg * (len(to_iter) - i - 1)
            # logger.info(f"Saved {i + 1} explanations. AVG s/it (tot): {tot_avg} (- {tot_remaining}) "
            #             f"| last AVG s/it: {checkpoint_avg} (- {remaining_checkpoint})")
            logger.info(
                f"Experiment it {i} completed in {actual_time - last_checkpoint_time} | AVG s/it: {tot_avg} (- {tot_remaining})"
                f" | last AVG s/it: {checkpoint_avg} (- {remaining_checkpoint})")
            last_checkpoint_time = actual_time
        end_time = datetime.now()
        logger.info(f"Experiment {experiment_id} completed in {end_time - start_time}")

    def launch_single_experiment(self, experiment_id, dataset_name, results_dir, random_seed, model_name,
                                 logger, explainer_name, dataset_params=None, model_params=None,
                                 explainer_params=None, **kwargs):
        dataset_params = copy.deepcopy(dataset_params) if dataset_params else {}
        model_params = copy.deepcopy(model_params) if model_params else {}
        explainer_params = copy.deepcopy(explainer_params) if explainer_params else {}
        try:
            if os.path.exists(results_dir):
                shutil.rmtree(results_dir + '.old', ignore_errors=True)
                shutil.move(results_dir, results_dir + '.old')

        except FileNotFoundError:
            pass
        os.makedirs(results_dir, exist_ok=True)
        base_experiment_params = dict(experiment_id=experiment_id, dataset_name=dataset_name,
                                      random_seed=random_seed, model_name=model_name, model_params=model_params,
                                      explainer_name=explainer_name, dataset_params=dataset_params,
                                      explainer_params=explainer_params)
        with open(os.path.join(results_dir, 'params.json'), 'w') as f:
            json.dump(base_experiment_params, f)
        for t_params in [model_params, explainer_params, dataset_params]:
            t_params['random_seed'] = t_params.get('random_seed', random_seed)
        dataset = xfc.datasets_loaders.dataset_loader_factory.create(dataset_name, **dataset_params)
        model = models.model_factory.create(model_name, **model_params)
        explainer = xfc.explainers.explainer_factory.create(explainer_name, **explainer_params)
        exp_list = []
        if hasattr(explainer, 'explain_list'):
            try:
                exp_list = explainer.explain_list(record_list=dataset, predict_method=model)
            except Exception as e:
                xfc.xfc_utils.handle_exception(e, logger)
        else:
            logger.info(f"Total records: {len(dataset)}")
            start_time = datetime.now()
            last_checkpoint_time = start_time
            checkpoint_each = 10
            for i, record in tqdm(enumerate(dataset)):
                try:
                    exp = explainer.explain(record=record, predictor=model)
                    exp_list.append(exp)
                except Exception as e:
                    logger.error(f"Error in record {i}")
                    xfc.xfc_utils.handle_exception(e, logger)

                if (i + 1) % checkpoint_each == 0:
                    with open(os.path.join(results_dir, 'explanation_list.pkl'), 'wb') as file:
                        pickle.dump(exp_list, file)
                if ((i + 1) % checkpoint_each == 0) or (i == 0) or (i == len(dataset) - 1):
                    actual_time = datetime.now()
                    tot_avg = (actual_time - start_time) / (i + 1)
                    tot_remaining = tot_avg * (len(dataset) - i - 1)

                    checkpoint_avg = (actual_time - last_checkpoint_time) / min(checkpoint_each, i + 1)
                    remaining_checkpoint = checkpoint_avg * (len(dataset) - i - 1)
                    logger.info(f"Saved {i + 1} explanations. AVG s/it (tot): {tot_avg} (- {tot_remaining}) "
                                f"| last AVG s/it: {checkpoint_avg} (- {remaining_checkpoint})")
                    last_checkpoint_time = actual_time
            logger.info(f"Saved {len(dataset)} explanations")
        with open(os.path.join(results_dir, 'explanation_list.pkl'), 'wb') as file:
            pickle.dump(exp_list, file)

        # remove result 'old' directory given that experiment is done
        if os.path.exists(results_dir + '.old'):
            shutil.rmtree(results_dir + '.old')


experiment_done = [
    'sk_f_jf_1.0',
    'sk_f_jf_1.1',
    'sk_f_jf_1.1b',
    'sk_f_jf_1.1n',
    'f_bs_1.0',
    'f_bs_1.1',

    # only claim
    'oc_1.0',
    'oc_1.1',
    'f_bs_1.1b',
    'f_bs_1.1c',

    # after submission EMNLP
    'fbs_np_1.0',
    'fbs_np_2.0',
    'oc_fbs_np_1.0',

    'fbs_time_2.0',
    'fbs_time_1.0',

    'fbs_time_1.1',
    'fbs_time_2.1',

    'lla_np_1.0',
    'lla_np_2.0',
    'lla_fv_1.0',
    'lla_fv_1.1',
    'lla_fv_1.2',

]

exp_to_analyse = [
    'fv_sf_1.0',
    'fv_sf_2.0',
    'fv_f2l_1.0',
    'fv_f2l_2.0',
    'fv_f3l_1.0',
    'fv_f3l_2.0',
    'r2_fv_f2l_1.0',
    'r2_fv_f2l_2.0',
    'r2_fv_f3l_1.0',
    'r2_fv_f3l_2.0',
    'r2_fv_sf_1.0',
    'r2_fv_sf_2.0',
    'fv_fm_1.0',
    'fv_fm_2.0',
    'r2_fv_fm_1.0',
    'r2_fv_fm_2.0',

    'fv_av_1.0',
    'fv_av_2.0',

    'gfce_f2l_1.0',
    'gfce_f2l_2.0',
    'gfce_f3l_1.0',  # OK 4:27h
    'gfce_fm2_1.0',  # OK 4:29h
    'gfce_sf_1.0',  # OK 1:10h
    'gfce_f3l_1.1',  # 4:49h
    'gfce_sf_2.1',  # 5:34h
    'gfce_sf_1.1',  # 1:10h
    'gfce_f3l_2.0',  # 22:25h
    'gfce_f3l_2.1',  # 22:52h

    'gfce_f3l_1.1F',  # 12:17h
    'gfce_sf_1.1F',  # 3:09h
    'gfce_av_1.0F',  # 8:00h
    'gfce_av_2.0',  # 13:55h
    'gfce_av_1.0',  # 2:55h

    'fv_f2lF_1.0',
    'fv_f2lF_2.0',
    'fv_f3lF_1.0',
    'fv_f3lF_2.0',

    'r2_fv_f2lF_1.0',
    'r2_fv_f2lF_2.0',
    'r2_fv_f3lF_1.0',
    'r2_fv_f3lF_2.0',

    'gfce_f2l_1.1',  # pepa 3
    'gfce_f2l_2.1',  # pepa 4
    'r2_fv_sf_1.0',
    'r2_fv_sf_2.0',
    'gfce_fm2_1.1',  # pepa 1
    'gfce_fm2_2.1',  # pepa 2

    'llama70b_f2l_1.0',  # 10:43h # bad version

    'llama70b_fm2_1.0',  # 18:40h
    'llama70b_fm2_2.0',
    'llama70b_sf_1.0',
    'llama70b_sf_2.0',
    'llama70b_av_1.0',
    'llama70b_av_2.0',
    'llama70b_f3l_1.0',
    'llama70b_f3l_2.0',  # 17:45h
    'llama70b_f2l_2.1',
    'llama70b_f2l_1.1',  # 8:17h
    'gfce_fm2_2.2',
    'gfce_fm2_1.2',
]

experiments_doing = [

    # ABB

    # ABC GPU 0

    # ABD GPU 0

    # ABE GPU 0
    # ABC GPU 0

    # ABF GPU 0

    # ABG GPU 2

]

test_conf = [
    'st_1.0',
    'lla_np_1.test',
    'sms_p_1.0',
]

# main thing to start the experiment
if __name__ == "__main__":

    experiments_to_run = [

        # 'fv_sf_1.0test',

        # 'fv_f2l_3.0',

        # 'st_1.1',
        # 'st_1.2',
        # 'lla_np_1.test',
    ]

    experiment_runner = ExperimentRunner()
    for exp_id in experiments_to_run:
        if exp_id in experiment_done:
            print(f"Experiment {exp_id} already done, skipping...")
            continue
        # if debug mode create additional params
        if xfc.xfc_utils.is_debugging():
            print(f"Running experiment {exp_id} in debug mode")
            additional_params = pydantic.utils.deep_update({'dataset_params': {'nrows': 5, 'skiprows': 100},
                                                            # 'model_params': {
                                                            #     'cache_pred_file': None,
                                                            # },

                                                            'explainer_params': {
                                                                'num_samples': [50]
                                                            },
                                                            },  # E.plain_pred_exp
                                                           )
        else:
            additional_params = None
        experiment_runner.launch_experiment_by_id(exp_id, additional_params=additional_params)
