import copy
import json
import os
import pickle

import numpy as np
from lime.lime_text import LimeTextExplainer

from transformers import PreTrainedTokenizer

from explainable_fact_checking.adapters.feverous_model import FeverousModelAdapter
from explanation_presentation import style_exp_to_html


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


class WrapperExplaniableFactChecking:
    example_structure = {
        "evidence": [{"content": ["Aramais Yepiskoposyan_cell_0_6_1", "FC Ararat Yerevan_sentence_1"], "context": {
            "Aramais Yepiskoposyan_cell_0_6_1": ["Aramais Yepiskoposyan_title",
                                                 "Aramais Yepiskoposyan_header_cell_0_6_0",
                                                 "Aramais Yepiskoposyan_header_cell_0_4_0"],
            "FC Ararat Yerevan_sentence_1": ["FC Ararat Yerevan_title"]}}], "id": 13969,
        "claim": "Aramais Yepiskoposyan played for FC Ararat Yerevan, an Armenian football club based in Yerevan during 1986 to 1991.",
        "label": "SUPPORTS", "annotator_operations": [{"operation": "start", "value": "start", "time": "0"},
                                                      {"operation": "Now on", "value": "?search=", "time": "0.573"},
                                                      {"operation": "search", "value": "Aramais Yepiskoposyan",
                                                       "time": "12.608"},
                                                      {"operation": "Now on", "value": "Aramais Yepiskoposyan",
                                                       "time": "13.316"}, {"operation": "Highlighting",
                                                                           "value": "Aramais Yepiskoposyan_cell_0_6_1",
                                                                           "time": "22.607"},
                                                      {"operation": "hyperlink", "value": "FC Ararat Yerevan",
                                                       "time": "26.241"},
                                                      {"operation": "Now on", "value": "FC Ararat Yerevan",
                                                       "time": "27.919"}, {"operation": "Highlighting",
                                                                           "value": "FC Ararat Yerevan_sentence_1",
                                                                           "time": "36.061"},
                                                      {"operation": "finish", "value": "finish", "time": "43.165"}],
        "challenge": "Multi-hop Reasoning",
        "input_txt_model": "Aramais Yepiskoposyan played for FC Ararat Yerevan, an Armenian football club based in Yerevan during 1986 to 1991. </s> Aramais Yepiskoposyan </s> 1986\u20131991 is [[FC_Ararat_Yerevan|FC Ararat Yerevan]]. </s> Senior career* is [[FC_Ararat_Yerevan|FC Ararat Yerevan]]. </s>  Football Club Ararat Yerevan ([[Armenian_language|Armenian]]: \u0556\u0578\u0582\u057f\u0562\u0578\u056c\u0561\u0575\u056b\u0576 \u0531\u056f\u0578\u0582\u0574\u0562 \u0531\u0580\u0561\u0580\u0561\u057f \u0535\u0580\u0587\u0561\u0576), commonly known as Ararat Yerevan, is an Armenian [[Association_football|football]] club based in [[Yerevan|Yerevan]] that plays in the Armenian Premier League.",
        "input_txt_to_use": "Aramais Yepiskoposyan played for FC Ararat Yerevan, an Armenian football club based in Yerevan during 1986 to 1991. </s> Aramais Yepiskoposyan </s> 1986\u20131991 is [[FC_Ararat_Yerevan|FC Ararat Yerevan]]. </s> Senior career* is [[FC_Ararat_Yerevan|FC Ararat Yerevan]]. </s>  Football Club Ararat Yerevan ([[Armenian_language|Armenian]]: \u0556\u0578\u0582\u057f\u0562\u0578\u056c\u0561\u0575\u056b\u0576 \u0531\u056f\u0578\u0582\u0574\u0562 \u0531\u0580\u0561\u0580\u0561\u057f \u0535\u0580\u0587\u0561\u0576), commonly known as Ararat Yerevan, is an Armenian [[Association_football|football]] club based in [[Yerevan|Yerevan]] that plays in the Armenian Premier League."}
    evidence_separator = r' </s> '
    reference_record = {}

    def __init__(self, record, predict_method, separator=r' </s> ', perturbation_mode='only_evidence', debug=False):
        self.record = record
        self.content_index_map = self.get_content_index_map(record)
        self.claim = None
        self.mode = 'normal'
        self.restructure_perturbed_records = self.restructure_perturbed_records_codes
        if 'claim' in record and 'input_txt_to_use' in record:
            assert record['claim'] == record['input_txt_to_use'].split(separator)[
                0], 'Claim and input_txt_to_use do not match'
            self.claim = record['claim']
            self.mode = 'input_txt_to_use'
            self.restructure_perturbed_records = self.restructure_perturbed_records_strings
        self.separator = separator
        self.predict_method = predict_method
        self.id = record['id']

        if perturbation_mode == 'only_evidence':
            self.get_text_to_perturb = self.get_evidence_string
        elif perturbation_mode == 'claim_and_evidence':
            # todo adjust with composition wrt the mode of perturbation
            raise(NotImplementedError('Not implemented yet'))


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
        pass

    def restructure_perturbed_records_strings(self, perturbed_evidence_string_list):
        """
        This method restructures the perturbed evidence list in the format of the example_structure
        it reconstructs the record with the claim using the input_txt_to_use field
        :param perturbed_evidence_string_list:
        :return:
        """
        perturbed_record_list = [copy.deepcopy(self.record) for _ in range(len(perturbed_evidence_string_list))]
        for i, (ev, turn_record) in enumerate(zip(perturbed_evidence_string_list, perturbed_record_list)):
            evidence_list = ev.split(self.separator)
            # remove 'UNKWORDZ' from the evidence list. lime with bow=False replace missing words with 'UNKWORDZ'
            evidence_list = [ev for ev in evidence_list if ev != 'UNKWORDZ']
            turn_record['id'] = i
            turn_record['input_txt_to_use'] = self.separator.join([self.claim] + evidence_list)
        return perturbed_record_list

    def restructure_perturbed_records_codes(self, perturbed_evidence_string_list):
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
        # if 'input_txt_to_use' in keys of dictionary split the string by '<\e>' the first item is the claim from the second to the end are the evidence
        # remove additional spaces
        # return the list of evidence
        if 'input_txt_to_use' in record:
            all_el = record['input_txt_to_use'].split(WrapperExplaniableFactChecking.evidence_separator)
            evidence_list = all_el[1:]
            claim = all_el[0]
            return [ev.strip() for ev in evidence_list]
        return record['evidence'][0]['content']



    @staticmethod
    def set_evidence_content(record, content):
        record['evidence'][0]['content'] = content


def explain_with_lime(file_to_explain, predictor, output_dir, num_samples, top=None, perturbation_mode='only_evidence'):
    data = []
    early_stop = top is not None
    with open(file_to_explain, 'r') as file:
        for i, line in enumerate(file):
            if early_stop and i >= top:
                break
            if line != '\n':
                data.append(json.loads(line))



    if predictor is None:
        fc_model = FeverousModelAdapter()
        predictor = fc_model.predict

    for record in data:
        xfc_wrapper = WrapperExplaniableFactChecking(record, predictor, debug=True, perturbation_mode=perturbation_mode)

        explainer = LimeTextExplainer(
            split_expression=xfc_wrapper.tokenizer,
            # Order matters, and we cannot use bag of words.
            bow=False,
            class_names=['NOT ENOUGH INFO', 'SUPPORTS', 'REFUTES'],
        )
        labels = range(len(explainer.class_names))

        exp = explainer.explain_instance(
            text_instance=xfc_wrapper.get_text_to_perturb(),
            classifier_fn=xfc_wrapper,
            labels=labels,
            num_features=xfc_wrapper.get_num_evidence(),
            num_samples=num_samples,
        )
        exp.claim = xfc_wrapper.claim
        exp.id = xfc_wrapper.get_id()
        exp.label = record['label']
        exp.record = record
        exp.num_samples = num_samples


        # 1 / (1 + np.exp(-predictions[0]))[:, 2] # NUMPY sigmoid
        exp_dir = os.path.join(output_dir, file_to_explain.split('/')[-1].split('.')[0])
        # delete the directory if it exists and recrete it

        os.makedirs(exp_dir, exist_ok=True)


        file_save_path = os.path.join(exp_dir, f'{xfc_wrapper.get_id()}')

        with open(file_save_path + '.html', 'w', encoding='utf-8') as file:
            file.write(style_exp_to_html(exp))
        #
        # with open(file_save_path + '_LIME.html', 'w', encoding='utf8') as file:
        #     file.write(exp.as_html())

        # save explanation in a file using python library pickle
        with open(file_save_path + '.pkl', 'wb') as file:
            pickle.dump(exp, file)

        # save explanation in a file using python library json
        exp_dict = {int(label): exp.as_list(label) for label in labels}
        exp_dict['intercept'] = exp.intercept
        exp_dict['claim'] = exp.claim
        exp_dict['class_names'] = exp.class_names


        with open(file_save_path + '.json', 'w') as file:
            json.dump(exp_dict, file)


# class to read jsonl files and predict the labels with the feverous model then takes the value of 'input_txt_to_use' for each prediction and add it to the record and save the new file
class AddInputTxtToUse:
    @staticmethod
    def predict_and_save(input_file, output_file, model):
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            record_list = [json.loads(line) for line in f_in]
            predictions = model.predict(record_list)
            full_predictions = model.predictions
            for record, pred in zip(record_list, full_predictions):
                record['input_txt_to_use'] = pred['input_txt_model']
                record['claim'] = pred['claim']

                f_out.write(json.dumps(record) + '\n')

def save_prediciton_without_evidence(input_file, output_file, model):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        record_list = [json.loads(line) for line in f_in]
        for record in record_list:
            record['evidence'] = []
        predictions = model.predict(record_list)
        full_predictions = model.predictions
        for record, pred in zip(record_list, full_predictions):
            record['input_txt_to_use'] = pred['input_txt_model']
            record['claim'] = pred['claim']
            record['evidence'] = []
            f_out.write(json.dumps(record) + '\n')