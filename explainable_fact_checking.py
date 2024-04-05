import copy
import json
import os
import pickle

import numpy as np
from lime.lime_text import LimeTextExplainer

from feverous_model import run_exp_multitest_wsavedmodel
from transformers import PreTrainedTokenizer


class CustomTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.vocab_file = vocab_file

        # Load vocabulary from file
        self.vocab = self.load_vocab(vocab_file)

    def load_vocab(self, vocab_file):
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = {word.strip(): i for i, word in enumerate(f)}
        return vocab

    def _tokenize(self, text):
        return text.split(" ||| ")

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        for token, idx in self.vocab.items():
            if idx == index:
                return token
        return self.unk_token


class FeverousModelAdapter:
    """
    Adapter for Feverous Model
    This class is an Adapter for the Feverous Model. It takes a claim and a set of evidence and returns a prediction.

    Initialized with a model.
    predict method takes a claim and a set of evidence in a in the following format:
    {"evidence": [{"content": ["ev_0", "ev_1",], "context": {"ev_0": ["metadata_0"], "ev_1": ["metadata_0", "metadata_1"] }}], "id": 20863, "claim": "Claim text", "label": "SUPPORTS | REFUTES | NOT ENOUGH INFO", "annotator_operations": [{"operation": "start", "value": "start", "time": "0"}, {"operation": "Now on", "value": "?search=", "time": "0.632"}, {"operation": "search", "value": "paul dicks", "time": "34.958"},  {"operation": "Highlighting", "value": "Paul Dicks_sentence_0", "time": "45.087"}, {"operation": "finish", "value": "finish", "time": "79.163"}], "challenge": "Combining Tables and Text"}
    save records in file in a temporary location in jsonl format with the format specified right above.
    Apply the model that loads the records and returns the prediction in a file.
    return predicitons loading the file.

    """

    def __init__(self):
        pass

    def predict(self, records):
        # Save records in file in jsonl format
        base_path = '/homes/bussotti/feverous_work/feverousdata/'
        tmp_file = 'AB/tmp_records.jsonl'
        # duplicate the first elemento of records list to avoid the error of the model
        records = [records[0]] + records

        with open(base_path + tmp_file, 'w') as file:
            for record in records:
                if record['label'] == 'NOT ENOUGH INFO':
                    record['label'] = 'REFUTES'
                file.write(json.dumps(record) + '\n')

        tmp_pred_name = 'AB/tmp_predictions'
        res = run_exp_multitest_wsavedmodel([tmp_pred_name], 'jf_home/feverous_wikiv1.db',
                                            'feverous_dev_challenges_sentencesandtable.jsonl',
                                            ['jf_home/feverous_wikiv1.db'],
                                            [tmp_file], '', None, None, True, True, 'models_fromjf270623or')

        # Prediciton format example
        # {   "claim": "Paul Dicks (born in 1950) was a Speaker of the Newfoundland and Labrador House of Assembly, preceding Thomas Lush and succeeded by Lloyd Snow.",
        #     "label": "SUPPORTS", "predicted_label": "SUPPORTS", "label_match": true}

        with open(base_path + tmp_pred_name + '_predictor.jsonl', 'r') as file:
            predictions = [json.loads(line) for line in file if line != '\n']

        return np.array([pred['predicted_scores'] for pred in predictions])


class WrapperExplaniableFactChecking:
    example_structure = {"evidence": [
        {"content": ["ev_0", "ev_1", ], "context": {"ev_0": ["metadata_0"], "ev_1": ["metadata_0", "metadata_1"]}}],
        "id": 20863, "claim": "Claim text", "label": "SUPPORTS | REFUTES | NOT ENOUGH INFO",
        "annotator_operations": [{"operation": "start", "value": "start", "time": "0"},
                                 {"operation": "Now on", "value": "?search=", "time": "0.632"},
                                 {"operation": "search", "value": "paul dicks", "time": "34.958"},
                                 {"operation": "Highlighting", "value": "Paul Dicks_sentence_0", "time": "45.087"},
                                 {"operation": "finish", "value": "finish", "time": "79.163"}],
        "challenge": "Combining Tables and Text"}

    reference_record = {}

    def __init__(self, record, predict_method, separator=r' ||| ', debug=False):
        self.record = record
        self.content_index_map = self.get_content_index_map(record)
        self.separator = separator
        self.predict_method = predict_method
        self.id = record['id']
        self.debug = debug
        if debug:
            self.debug_data = {}

    def get_id(self):
        return self.id

    def get_num_evidence(self):
        return len(self.get_evidence_list(self.record))

    @staticmethod
    def get_content_index_map(record):
        evidence_content = WrapperExplaniableFactChecking.get_evidence_list(record)
        content_index_map = {}
        for i, ev in enumerate(evidence_content):
            content_index_map[ev] = i
        return content_index_map

    def get_evidence_string(self):
        evidence_content = self.get_evidence_list(self.record)
        return self.separator.join(evidence_content)

    def predict_wrapper(self, perturbed_evidence_string_list):
        restructured_records = self.restructure_perturbed_records(perturbed_evidence_string_list)
        predictions = self.predict_method(restructured_records)

        if self.debug:
            self.debug_data['perturbed_evidence_string_list'] = perturbed_evidence_string_list
            self.debug_data['restructured_records'] = restructured_records
            self.debug_data['predictions'] = predictions
            assert len(perturbed_evidence_string_list) == len(restructured_records) == len(predictions)

        return predictions

    def __call__(self, arg):
        return self.predict_wrapper(arg)

    def tokenizer(self, text):
        return text.split(self.separator)

    def restructure_perturbed_records(self, perturbed_evidence_string_list):
        """

        :param perturbed_evidence_list: a list of perturbed evidence
        :return: a list of records with the same structure of the example_structure
        """
        perturbed_record_list = [copy.deepcopy(self.record) for _ in range(len(perturbed_evidence_string_list))]
        for i, (ev, turn_record) in enumerate(zip(perturbed_evidence_string_list, perturbed_record_list)):
            evidence_list = ev.split(self.separator)
            # remove 'UNKWORDZ' from the evidence list. lime with bow=False replace missing words with 'UNKWORDZ'
            evidence_list = [ev for ev in evidence_list if ev != 'UNKWORDZ']
            turn_record['id'] = i
            self.set_evidence_content(turn_record, evidence_list)
        if self.debug:
            assert not np.all(
                [perturbed_record_list[0]['evidence'] == x['evidence'] for x in perturbed_record_list[1:]])

        # [x['evidence'] for x in perturbed_record_list]
        return perturbed_record_list

    @staticmethod
    def get_evidence_list(record):
        return record['evidence'][0]['content']

    @staticmethod
    def set_evidence_content(record, content):
        record['evidence'][0]['content'] = content


def explain_with_lime(file_to_explain, predictor=None):
    AB_path = '/homes/bussotti/feverous_work/feverousdata/AB/'
    example_file = os.path.join('/homes/bussotti/feverous_work/feverousdata/', file_to_explain)

    with open(example_file, 'r') as file:
        data = [json.loads(line) for line in file if line != '\n']

    if predictor is None:
        fc_model = FeverousModelAdapter()
        predictor = fc_model.predict
    for record in data:
        xfc_wrapper = WrapperExplaniableFactChecking(record, predictor, debug=True)

        explainer = LimeTextExplainer(
            split_expression=xfc_wrapper.tokenizer,
            # Order matters, and we cannot use bag of words.
            bow=False,
            class_names=['NOT ENOUGH INFO', 'SUPPORTS', 'REFUTES'],
        )
        labels = range(len(explainer.class_names))
        exp = explainer.explain_instance(
            text_instance=xfc_wrapper.get_evidence_string(),
            classifier_fn=xfc_wrapper,
            labels=labels,
            num_features=xfc_wrapper.get_num_evidence(),
            num_samples=50,
        )
        # 1 / (1 + np.exp(-predictions[0]))[:, 2] # NUMPY sigmoid
        exp_dir = os.path.join(AB_path, 'lime_explanations', file_to_explain)
        os.makedirs(exp_dir, exist_ok=True)
        file_save_path = os.path.join(exp_dir, f'{xfc_wrapper.get_id()}')

        exp.save_to_file(file_path=file_save_path + '.html')
        # save explanation in a file using python library pickle
        with open(file_save_path + '.pkl', 'wb') as file:
            pickle.dump(exp, file)

        # save explanation in a file using python library json
        exp_dict = {int(label): exp.as_list(label) for label in labels}
        exp_dict['intercept'] = exp.intercept

        with open(file_save_path + '.json', 'w') as file:
            json.dump(exp_dict, file)
