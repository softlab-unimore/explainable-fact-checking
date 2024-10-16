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

import explainable_fact_checking as xfc
import explainable_fact_checking.experiment_definitions


class LimeXFCAdapter:
    def __init__(self, perturbation_mode, num_samples=50, random_seed=42, wrapper_params=None, class_names=None,
                 ):
        if wrapper_params is None:
            wrapper_params = {}
        if class_names is None:
            class_names = explainable_fact_checking.experiment_definitions.E.CLASS_NAMES_V0
        self.num_samples = num_samples
        self.perturbation_mode = perturbation_mode
        self.random_seed = random_seed
        self.wrapper_params = wrapper_params
        self.class_names = class_names

    def __call__(self, *args, **kwargs):
        return self.explain(*args, **kwargs)

    def explain(self, record, predictor):
        xfc_wrapper = xfc.wrappers.ModelWrapper(record, predictor, debug=True,
                                                perturbation_mode=self.perturbation_mode, **self.wrapper_params)

        start_time = datetime.now()
        explainer = LimeTextExplainer(
            split_expression=xfc_wrapper.tokenizer,
            # Order matters, and we cannot use bag of words.
            bow=False,
            class_names=self.class_names,
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
        exp.execution_time = (b - start_time).total_seconds()
        exp.params_to_report = xfc_wrapper.params_to_report
        exp.explanation_time = (b - a).total_seconds()
        return exp


class ShapXFCAdapter:
    def __init__(self, perturbation_mode, mode='KernelExplainer', num_samples=50, random_seed=42, wrapper_params=None,
                 class_names=None):
        if class_names is None:
            class_names = explainable_fact_checking.experiment_definitions.E.CLASS_NAMES_V0
        if wrapper_params is None:
            wrapper_params = {}
        self.num_samples = num_samples
        self.perturbation_mode = perturbation_mode
        self.random_seed = random_seed
        self.mode = mode
        self.wrapper_params = wrapper_params
        self.class_names = class_names

    def __call__(self, *args, **kwargs):
        return self.explain(*args, **kwargs)

    def explain(self, record, predictor):
        xfc_wrapper = xfc.wrappers.ModelWrapper(record, predictor=predictor, debug=True,
                                                perturbation_mode=self.perturbation_mode,
                                                explainer='shap', **self.wrapper_params)
        evidence_array = xfc_wrapper.get_evidence_list_SHAP()
        # set numpy random state
        np.random.seed(self.random_seed)
        start_time = datetime.now()
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
                silent=True,
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
        tot_time = (b - start_time).total_seconds()
        explanation_time = (b - a).total_seconds()
        # convert the explanation to the same format of LIME
        local_exp = {i: [(j, shap_values[j]) for j in range(len(shap_values))] for i, shap_values in
                     enumerate(exp[0].T)}
        indexed_string = lime.lime_text.IndexedString(raw_string=xfc_wrapper.get_text_to_perturb(),
                                                      split_expression=xfc_wrapper.tokenizer, bow=False, )
        domain_mapper = lime.lime_text.TextDomainMapper(indexed_string=indexed_string)
        predict_proba = xfc_wrapper.predict_wrapper(np.ones(evidence_array.shape)).reshape(-1)
        # predict_proba = predictor([record])
        # Explanation(
        #     exp,
        #     base_values=predict_proba,
        #     data=X.to_numpy() if isinstance(X, pd.DataFrame) else X,
        #     feature_names=xfc_wrapper.evidence_array,
        #     compute_time=tot_time,
        # )
        exp_dict = dict(
            local_exp=local_exp,
            claim=xfc_wrapper.claim,
            id=xfc_wrapper.get_id(),
            label=record['label'],
            class_names=self.class_names,
            record=record,
            execution_time=explanation_time,
            predict_proba=predict_proba,
            domain_mapper=domain_mapper,
            random_seed=self.random_seed,
            explainer='SHAP',
            mode=self.mode,
            intercept=predict_proba - np.sum(exp[0], axis=0),
            params_to_report=xfc_wrapper.params_to_report,
            explanation_time=explanation_time,
        )
        return exp_dict


class OnlyClaimPredictorFeverous:
    def __init__(self, random_seed=42, class_names=None):
        self.random_seed = random_seed
        if class_names is None:
            raise ValueError("class_names must be provided")
        self.class_names = class_names

    def explain_list(self, record_list, predict_method):
        for record in record_list:
            record['evidence'] = []
            # delete the input_txt_to_use field if present
            if 'input_txt_to_use' in record:
                del record['input_txt_to_use']
        predictions = predict_method(record_list)
        full_predictions = predict_method.predictions
        for record, pred in zip(record_list, full_predictions):
            record['input_txt_to_use'] = pred['input_txt_model']
            record['claim'] = pred['claim']
            record['intercept'] = record['predict_proba'] = pred['predicted_scores']
        return record_list



class OnlyClaimPredictor:
    def __init__(self, random_seed=42, class_names=None):
        self.random_seed = random_seed
        if class_names is None:
            raise ValueError("class_names must be provided")
        self.class_names = class_names

    def explain_list(self, record_list, predict_method):
        for record in record_list:
            record['evidence'] = []
        predictions = predict_method(record_list)
        for record, pred in zip(record_list, predictions):
            record['intercept'] = record['predict_proba'] = pred['predicted_scores']
        return record_list

class PlainPrediction:
    def __init__(self, random_seed=42, class_names=None):
        self.random_seed = random_seed
        if class_names is None:
            raise ValueError("class_names must be provided")
        self.class_names = class_names

    def explain_list(self, record_list, predict_method):
        predictions = predict_method(record_list)
        for record, pred in zip(record_list, predictions):
            record['intercept'] = record['predict_proba'] = pred['predicted_scores']
        return record_list


explainer_factory = xfc.xfc_utils.GeneralFactory()

# Register the explainers
explainer_factory.register_creator('lime', LimeXFCAdapter)
explainer_factory.register_creator('shap', ShapXFCAdapter)
explainer_factory.register_creator('claim_only_pred', OnlyClaimPredictorFeverous)
explainer_factory.register_creator('claim_only_pred_general', OnlyClaimPredictor)
explainer_factory.register_creator('plain_pred', PlainPrediction)


