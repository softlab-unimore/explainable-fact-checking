import gc
import itertools
import json
import pickle
from collections.abc import Iterable
from datetime import datetime

import send2trash
from tqdm import tqdm

import explainable_fact_checking as xfc
import os



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
        for i, kwargs in tqdm(enumerate(self.product_dict(**params_dict_to_iterate))):
            gc.collect()
            turn_a = datetime.now()
            logger.info(f"Iteration: {i} with parameters: {kwargs}")
            kwargs['results_dir'] = os.path.join(results_dir, str(i))
            self.launch_single_experiment(**kwargs, logger=logger)
            turn_b = datetime.now()
            logger.info(f"Iteration: {i} completed in {turn_b - turn_a}")



    def launch_single_experiment(self, experiment_id, dataset_name, results_dir, random_seed, model_name, model_params,
                                 explainer_name, dataset_params,
                                 explainer_params, logger, **kwargs):
        try:
            os.makedirs(results_dir, exist_ok=True)
            for filepath in os.scandir(results_dir):
                send2trash.send2trash(filepath)
        except FileNotFoundError:
            pass

        base_experiment_params = dict(experiment_id=experiment_id, dataset_name=dataset_name,
                                      random_seed=random_seed, model_name=model_name, model_params=model_params,
                                      explainer_name=explainer_name, dataset_params=dataset_params,
                                      explainer_params=explainer_params)
        with open(os.path.join(results_dir, 'params.json'), 'w') as f:
            json.dump(base_experiment_params, f)

        dataset = xfc.datasets_loaders.dataset_loader_factory.create(dataset_name, **dataset_params)
        model = xfc.models.model_factory.create(model_name, random_state=random_seed, **model_params)
        explainer = xfc.explainers.explainer_factory.create(explainer_name, **explainer_params) # todo
        exp_list = []
        for i, record in tqdm(enumerate(dataset)):
            exp = explainer.explain(record=record, predictor=model.predict)
            exp_list.append(exp)
            if i % 10 == 0:
                with open(os.path.join(results_dir, 'explanation_list.pkl'), 'wb') as file:
                    pickle.dump(exp_list, file)
        with open(os.path.join(results_dir, 'explanation_list.pkl'), 'wb') as file:
            pickle.dump(exp_list, file)


# main thing to start the experiment
if __name__ == "__main__":
    experiment_done = [
        'sk_f_jf_1.0',
    ]

    experiments_to_run = [

        'sk_f_jf_1.1',
        ]

    experiment_runner = ExperimentRunner()
    for exp_id in experiments_to_run:
        if exp_id in experiment_done:
            continue
        exp_dict = xfc.experiment_definitions.get_config_by_id(exp_id)
        experiment_runner.launch_experiment_by_config(exp_dict)

    # fc_model = FeverousModelAdapter()
    # predictor = fc_model.predict
    # for name in [
    #     'original_TO_01_formatted.jsonl',
    #     # 'ex_AB_00.jsonl',
    #     # 'feverous_dev_ST_01.jsonl',
    #     # 'feverous_dev_SO_01.jsonl',
    # ]:
    #     input_file_path = os.path.join('/homes/bussotti/feverous_work/feverousdata/AB/', name)
    #
    #     explainable_fact_checking.explainers.explain_with_SHAP(input_file_path, predictor=predictor,
    #                                                            output_dir='/homes/bussotti/feverous_work/feverousdata/AB/SHAP_explanations',
    #                                                            num_samples=500,
    #                                                            top=1000, mode='KernelExplainer')
    #
    #     # xfc.wrappers.explain_with_lime(input_file_path, predictor=predictor,
    #     #                   output_dir='/homes/bussotti/feverous_work/feverousdata/AB/lime_explanations', num_samples=500,
    #     #                   top=1000)
