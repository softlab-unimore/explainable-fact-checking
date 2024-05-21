import json
import os
import pickle
from datetime import datetime

import lime.lime_text
import numpy as np
import pandas as pd
import shap

from lime.lime_text import LimeTextExplainer
from shap import Explanation
from typing_extensions import deprecated

import explainable_fact_checking as xfc



class LimeXFCAdapter:
    def __init__(self,perturbation_mode, num_samples=50, random_seed=42, **kwargs):
        self.kwargs = kwargs
        self.num_samples = num_samples
        self.perturbation_mode = perturbation_mode
        self.random_seed = random_seed


    def __call__(self, *args, **kwargs):
        return self.explain(*args, **kwargs)

    def explain(self, record, predictor):
        xfc_wrapper = xfc.wrappers.FeverousRecordWrapper(record, predictor, debug=True, perturbation_mode=self.perturbation_mode)

        explainer = LimeTextExplainer(
            split_expression=xfc_wrapper.tokenizer,
            # Order matters, and we cannot use bag of words.
            bow=False,
            class_names=['NOT ENOUGH INFO', 'SUPPORTS', 'REFUTES'],
            random_state=self.random_seed,
        )
        labels = range(len(explainer.class_names))

        # time the explanation process
        a = datetime.now()
        exp = explainer.explain_instance(
            text_instance=xfc_wrapper.get_text_to_perturb(),
            classifier_fn=xfc_wrapper,
            labels=labels,
            num_features=xfc_wrapper.get_num_evidence(),
            num_samples=self.num_samples,
        )
        b = datetime.now()
        exp.claim = xfc_wrapper.claim
        exp.id = xfc_wrapper.get_id()
        exp.label = record['label']
        exp.record = record
        exp.execution_time = (b - a).total_seconds()
        return exp


def explain_with_lime(self, file_to_explain, predictor, output_dir, num_samples, top=None, perturbation_mode='only_evidence',
                      random_seed=42):
    data = []
    early_stop = top is not None
    with open(file_to_explain, 'r') as file:
        for i, line in enumerate(file):
            if early_stop and i >= top:
                break
            if line != '\n':
                data.append(json.loads(line))

    if predictor is None:
        fc_model = xfc.FeverousModelAdapter()
        predictor = fc_model.predict
    exp_dir = os.path.join(output_dir, file_to_explain.split('/')[-1].split('.')[0])
    os.makedirs(exp_dir, exist_ok=True)

    logger = xfc.xfc_utils.init_logger(exp_dir)

    for record in data:
        xfc_wrapper = xfc.wrappers.FeverousRecordWrapper(record, predictor, debug=True, perturbation_mode=perturbation_mode)

        explainer = LimeTextExplainer(
            split_expression=xfc_wrapper.tokenizer,
            # Order matters, and we cannot use bag of words.
            bow=False,
            class_names=['NOT ENOUGH INFO', 'SUPPORTS', 'REFUTES'],
            random_state=random_seed,
        )
        labels = range(len(explainer.class_names))

        # time the explanation process
        a = datetime.now()
        exp = explainer.explain_instance(
            text_instance=xfc_wrapper.get_text_to_perturb(),
            classifier_fn=xfc_wrapper,
            labels=labels,
            num_features=xfc_wrapper.get_num_evidence(),
            num_samples=num_samples,
        )
        b = datetime.now()
        exp.claim = xfc_wrapper.claim
        exp.id = xfc_wrapper.get_id()
        exp.label = record['label']

        exp.record = record
        exp.num_samples = num_samples
        exp.execution_time = (b - a).total_seconds()
        exp.filename = file_to_explain
        exp.random_seed = random_seed

        # 1 / (1 + np.exp(-predictions[0]))[:, 2] # NUMPY sigmoid

        file_save_path = os.path.join(exp_dir, f'{xfc_wrapper.get_id()}')
        # save explanation in a file using python library pickle
        with open(file_save_path + '.pkl', 'wb') as file:
            pickle.dump(exp, file)

        # read the explanation from the file debug script
        # with open(file_save_path + '.pkl', 'rb') as file:
        #     exp = pickle.load(file)

        # with open(file_save_path + '.html', 'w', encoding='utf-8') as file:
        #     file.write(style_exp_to_html(exp))
        #
        # with open(file_save_path + '_LIME.html', 'w', encoding='utf8') as file:
        #     file.write(exp.as_html())

        # save explanation in a file using python library json
        # exp_dict = {int(label): exp.as_list(label) for label in labels}
        # exp_dict['intercept'] = exp.intercept
        # exp_dict['claim'] = exp.claim
        # exp_dict['class_names'] = exp.class_names

        # with open(file_save_path + '.json', 'w') as file:
        #     json.dump(exp_dict, file)



class ShapXFCAdapter:
    def __init__(self, perturbation_mode, mode='KernelExplainer', num_samples=50, random_seed=42, **kwargs):
        self.kwargs = kwargs
        self.num_samples = num_samples
        self.perturbation_mode = perturbation_mode
        self.random_seed = random_seed
        self.mode = mode

    def __call__(self, *args, **kwargs):
        return self.explain(*args, **kwargs)

    def explain(self, record, predictor):
        xfc_wrapper = xfc.wrappers.FeverousRecordWrapper(record, predictor, debug=True, perturbation_mode=self.perturbation_mode,
                                            explainer='shap')
        evidence_array = xfc_wrapper.get_evidence_list_SHAP()
        # set numpy random state
        np.random.seed(self.random_seed)
        if self.mode == 'KernelExplainer':
            explainer = shap.KernelExplainer(model=xfc_wrapper, data=np.ones((1, evidence_array.shape[1])) * -1)
            # explainer = shap.KernelExplainer(model=xfc_wrapper,
            #                                  data=np.arange(evidence_array.shape[1]).reshape(1, -1) + 1)
            # time the explanation process
            a = datetime.now()
            # exp = explainer.shap_values(
            #     np.zeros(evidence_array.shape),
            #     nsamples=num_samples if num_samples is not None else "auto",
            # )
            X = np.arange(evidence_array.shape[1]).reshape(1, -1)
            exp = explainer.shap_values(
                X,
                nsamples=self.num_samples if self.num_samples is not None else "auto",
            )
            b = datetime.now()
        elif self.mode == 'Permutation':
            explainer = shap.PermutationExplainer(model=xfc_wrapper,
                                                  masker=np.arange(evidence_array.shape[1]).reshape(1, -1) + 1)
            # time the explanation process
            X = np.arange(evidence_array.shape[1]).reshape(1, -1)
            a = datetime.now()
            exp = explainer(
                X,
                max_evals=self.num_samples if self.num_samples is not None else "auto",
                main_effects=True
            )
            b = datetime.now()
        else:
            raise ValueError(f"Invalid mode {self.mode}")
        time_elapsed = (b - a).total_seconds()
        # convert the explanation to the same format of LIME
        local_exp = {i: [(j, shap_values[j]) for j in range(len(shap_values))] for i, shap_values in enumerate(exp.T)}
        indexed_string = lime.lime_text.IndexedString(raw_string=xfc_wrapper.get_text_to_perturb(),
                                                      split_expression=xfc_wrapper.tokenizer, bow=False, )
        domain_mapper = lime.lime_text.TextDomainMapper(indexed_string=indexed_string)
        predict_proba = xfc_wrapper.predict_wrapper(np.ones(evidence_array.shape)).reshape(-1)
        # predict_proba = predictor([record])
        Explanation(
            exp,
            base_values=predict_proba,
            data=X.to_numpy() if isinstance(X, pd.DataFrame) else X,
            feature_names=xfc_wrapper.evidence_array,
            compute_time=time_elapsed,
        )
        exp_dict = dict(
            local_exp=local_exp,
            claim=xfc_wrapper.claim,
            id=xfc_wrapper.get_id(),
            label=record['label'],
            record=record,
            num_samples=self.num_samples,
            execution_time=time_elapsed,
            predict_proba=predict_proba,
            domain_mapper=domain_mapper,
            random_seed=self.random_seed,
            explainer='SHAP',
            mode=self.mode,
        )
        return exp_dict

def explain_with_SHAP(file_to_explain, predictor, output_dir, num_samples, top=None, perturbation_mode='only_evidence',
                      random_seed=42, model_name=None, mode='Permutation'):
    data = []
    early_stop = top is not None
    with open(file_to_explain, 'r') as file:
        for i, line in enumerate(file):
            if early_stop and i >= top:
                break
            if line != '\n':
                data.append(json.loads(line))

    if predictor is None:
        fc_model = xfc.wrappers.FeverousModelAdapter()
        predictor = fc_model.predict

    for record in data:
        xfc_wrapper = xfc.wrappers.FeverousRecordWrapper(record, predictor, debug=True, perturbation_mode=perturbation_mode,
                                            explainer='shap')
        evidence_array = xfc_wrapper.get_evidence_list_SHAP()
        # set numpy random state
        np.random.seed(random_seed)
        if mode == 'KernelExplainer':
            explainer = shap.KernelExplainer(model=xfc_wrapper, data=np.ones((1, evidence_array.shape[1])) * -1)
            # explainer = shap.KernelExplainer(model=xfc_wrapper,
            #                                  data=np.arange(evidence_array.shape[1]).reshape(1, -1) + 1)
            # time the explanation process
            a = datetime.now()
            # exp = explainer.shap_values(
            #     np.zeros(evidence_array.shape),
            #     nsamples=num_samples if num_samples is not None else "auto",
            # )
            X = np.arange(evidence_array.shape[1]).reshape(1, -1)
            exp = explainer.shap_values(
                X,
                nsamples=num_samples if num_samples is not None else "auto",
            )
            b = datetime.now()
        elif mode == 'Permutation':
            explainer = shap.PermutationExplainer(model=xfc_wrapper,
                                                  masker=np.arange(evidence_array.shape[1]).reshape(1, -1) + 1)
            # time the explanation process
            X = np.arange(evidence_array.shape[1]).reshape(1, -1)
            a = datetime.now()
            exp = explainer(
                X,
                max_evals=num_samples if num_samples is not None else "auto",
                main_effects=True
            )
            b = datetime.now()
        time_elapsed = (b - a).total_seconds()
        # convert the explanation to the same format of LIME
        local_exp = {i: [(j, shap_values[j]) for j in range(len(shap_values))] for i, shap_values in enumerate(exp.T)}
        indexed_string = lime.lime_text.IndexedString(raw_string=xfc_wrapper.get_text_to_perturb(),
                                                      split_expression=xfc_wrapper.tokenizer, bow=False, )
        domain_mapper = lime.lime_text.TextDomainMapper(indexed_string=indexed_string)
        predict_proba = xfc_wrapper.predict_wrapper(np.ones(evidence_array.shape)).reshape(-1)
        # predict_proba = predictor([record])
        Explanation(
            exp,
            base_values=predict_proba,
            data=X.to_numpy() if isinstance(X, pd.DataFrame) else X,
            feature_names=xfc_wrapper.evidence_array,
            compute_time=time_elapsed,
        )
        exp_dict = dict(
            local_exp=local_exp,
            claim=xfc_wrapper.claim,
            id=xfc_wrapper.get_id(),
            label=record['label'],
            record=record,
            num_samples=num_samples,
            execution_time=time_elapsed,
            filename=file_to_explain,
            predict_proba=predict_proba,
            domain_mapper=domain_mapper,
            random_seed=random_seed,
            model_name=model_name,
            explainer='SHAP',
            mode=mode,
        )

        # 1 / (1 + np.exp(-predictions[0]))[:, 2] # NUMPY sigmoid
        exp_dir = os.path.join(output_dir, file_to_explain.split('/')[-1].split('.')[0])
        # delete the directory if it exists and recrete it

        os.makedirs(exp_dir, exist_ok=True)

        file_save_path = os.path.join(exp_dir, f'{xfc_wrapper.get_id()}')
        # save explanation in a file using python library pickle
        with open(file_save_path + '.pkl', 'wb') as file:
            pickle.dump(exp_dict, file)




explainer_factory = xfc.xfc_utils.GeneralFactory()

# Register the explainers
explainer_factory.register_creator('lime', LimeXFCAdapter)
explainer_factory.register_creator('shap', ShapXFCAdapter)
