import gc
import itertools
import json
import pickle
import shutil
import sys
from collections.abc import Iterable
from datetime import datetime

from tqdm import tqdm

import explainable_fact_checking as xfc
import os

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA_DEVICE_ORDER:", os.environ.get("CUDA_DEVICE_ORDER"))
import torch

print("CUDA is available:", torch.cuda.is_available())
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Root directory
base_path = '/homes/bussotti/feverous_work/feverousdata'


class ExperimentRunner:

    @staticmethod
    def product_dict(**kwargs):
        keys = kwargs.keys()
        return [dict(zip(keys, instance)) for instance in itertools.product(*kwargs.values())]

    def launch_experiment_by_config(self, exp_dict: dict):
        if exp_dict is None:
            raise ValueError(f"{exp_dict} is not a valid experiment id")
        experiment_id: str = exp_dict.get('experiment_id')
        if not isinstance(experiment_id, str):
            raise ValueError('You must specify an experiment id')
        for attr in xfc.experiment_definitions.REQUIRED_FIELDS:
            if attr not in exp_dict.keys():
                raise ValueError(f'You must specify some value for {attr} parameter. It\'s empty.')

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
                    # if str convert to list
                    if isinstance(v, str):
                        value[k] = [v]
                    if not isinstance(v, list) and not isinstance(v, Iterable):
                        raise TypeError(f"Value {v} is not a valid collection")
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
        logger = xfc.xfc_utils.init_logger(save_dir=results_dir)
        logger.info(f"Starting experiment {experiment_id}")
        start_time = datetime.now()
        for i, kwargs in tqdm(enumerate(self.product_dict(**params_dict_to_iterate))):
            gc.collect()
            logger.info(f"Iteration: {i} with parameters: {kwargs}")
            kwargs['results_dir'] = os.path.join(results_dir, str(i))
            turn_a = datetime.now()

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
            turn_b = datetime.now()
            logger.info(f"Iteration: {i} completed in {turn_b - turn_a}")
        end_time = datetime.now()
        logger.info(f"Experiment {experiment_id} completed in {end_time - start_time}")

    def launch_single_experiment(self, experiment_id, dataset_name, results_dir, random_seed, model_name, model_params,
                                 explainer_name, dataset_params,
                                 explainer_params, logger, **kwargs):
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

        dataset = xfc.datasets_loaders.dataset_loader_factory.create(dataset_name, **dataset_params)
        model = xfc.models.model_factory.create(model_name, random_state=random_seed, **model_params)
        explainer = xfc.explainers.explainer_factory.create(explainer_name, **explainer_params)
        exp_list = []
        if hasattr(explainer, 'explain_list'):
            try:
                exp_list = explainer.explain_list(record_list=dataset, predict_method=model)
            except Exception as e:
                xfc.xfc_utils.handle_exception(e, logger)
        else:
            for i, record in tqdm(enumerate(dataset)):
                try:
                    exp = explainer.explain(record=record, predictor=model)
                    exp_list.append(exp)
                except Exception as e:
                    logger.error(f"Error in record {i}")
                    xfc.xfc_utils.handle_exception(e, logger)

                if i % 10 == 0:
                    with open(os.path.join(results_dir, 'explanation_list.pkl'), 'wb') as file:
                        pickle.dump(exp_list, file)
        with open(os.path.join(results_dir, 'explanation_list.pkl'), 'wb') as file:
            pickle.dump(exp_list, file)

        # remove result 'old' directory given that experiment is done
        if os.path.exists(results_dir + '.old'):
            shutil.rmtree(results_dir + '.old')


experiment_done = [

]

experiments_doing = [
    'sk_f_jf_1.0',
    'sk_f_jf_1.1',
    'sk_f_jf_1.1b',
    'sk_f_jf_1.1n',
    'f_bs_1.0',
    'f_bs_1.1',

    # only claim
    'oc_1.0',
    'oc_1.1',
]

# main thing to start the experiment
if __name__ == "__main__":

    experiments_to_run = [

    ]

    experiment_runner = ExperimentRunner()
    for exp_id in experiments_to_run:
        if exp_id in experiment_done:
            print(f"Experiment {exp_id} already done, skipping...")
            continue
        exp_dict = xfc.experiment_definitions.get_config_by_id(exp_id)
        experiment_runner.launch_experiment_by_config(exp_dict)
