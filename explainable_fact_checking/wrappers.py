import copy

import numpy as np
from transformers import PreTrainedTokenizer
import explainable_fact_checking as xfc
from explainable_fact_checking import C


class CustomTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file, *args, **kwargs):
        super().__init__(**kwargs)
        self.vocab_file = vocab_file

        # Load vocabulary from file
        self.vocab = self.load_vocab(vocab_file)

    def load_vocab(self, vocab_file):
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = {word.strip(): i for i, word in enumerate(f)}
        return vocab

    def _tokenize(self, text, **kwargs):
        return text.split(" ||| ")

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        for token, idx in self.vocab.items():
            if idx == index:
                return token
        return self.unk_token


class ModelWrapper:
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


    def __init__(self, record, predictor, separator=r' </s> ', perturbation_mode='only_evidence', explainer='lime', debug=False):
        self.record = record
        self.separator = separator
        self.predict_method = predictor
        self.explainer = explainer
        self.validate()
        self.id = record['id']
        self.claim = record['claim']
        self.evidence_array = np.array(self.generate_evidence_list(record))
        self.evidence_index_map = self.generate_evidence_index_map(self.evidence_array)
        self.mode = self.determine_mode()
        self.params_to_report = {}
        self.debug = debug
        if self.debug:
            self.debug_data = {}

        if self.mode == C.EV_KEY:
            self.restructure_records = self.restructure_records_strings
        elif self.mode == 'normal':
            self.restructure_records = self.restructure_records_dict

        if explainer == 'shap':
            self.reconstruct_evidence_list = self.reconstruct_from_shap
        elif explainer == 'lime':
            self.reconstruct_evidence_list = self.reconstruct_from_lime

        if perturbation_mode == 'only_evidence':
            self.get_text_to_perturb = self.get_evidence_string
        elif perturbation_mode == 'claim_and_evidence':
            self.get_text_to_perturb = self.raise_not_implemented

    def validate(self):
        if 'claim' not in self.record:
            raise ValueError('Claim not found in record')
        if 'id' not in self.record:
            raise ValueError('Id not found in record')
        if self.explainer not in ['shap', 'lime']:
            raise ValueError('Explainer not recognized. Choose between shap and lime.')

    def determine_mode(self):
        record = self.record
        if C.TXT_TO_USE in record:
            assert self.claim == record[C.TXT_TO_USE].split(self.separator)[0], 'Claim and input_txt_to_use do not match'
        elif C.EV_KEY in record:
            return C.EV_KEY
        return 'normal'

    def select_text_perturbation(self, perturbation_mode):
        modes = {
            'only_evidence': self.get_evidence_string,
            'claim_and_evidence': self.raise_not_implemented
        }
        return modes.get(perturbation_mode, self.raise_not_implemented)

    def raise_not_implemented(self):
        raise NotImplementedError('Not implemented yet')

    def get_id(self):
        return self.id

    def get_num_evidence(self):
        return len(self.generate_evidence_list(self.record))

    @staticmethod
    def generate_evidence_index_map(evidence_list):
        return {ev: i for i, ev in enumerate(evidence_list)}

    def get_evidence_string(self):
        evidence_content = self.generate_evidence_list(self.record)
        return self.separator.join(evidence_content)

    def get_evidence_list_SHAP(self):
        evidence_list = self.generate_evidence_list(self.record)
        return np.array(evidence_list).reshape(1, -1)

    def predict_wrapper(self, perturbed_evidence_string_list):
        restructured_records = self.restructure_records(perturbed_evidence_string_list)
        predictions = self.predict_method(restructured_records)
        n_samples = len(restructured_records)
        self.params_to_report |= {
            'total_predictions': self.params_to_report.get('total_predictions', 0) + n_samples,
        }
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

    def masker(self, *args):
        return np.ones((1, len(args)))

    def restructure_perturbed_records(self, perturbed_evidence_string_list):
        pass

    def restructure_records_dict(self, perturbed_item_list):
        perturbed_record_list = [copy.deepcopy(self.record) for _ in range(len(perturbed_item_list))]
        for i, (ev, turn_record) in enumerate(zip(perturbed_item_list, perturbed_record_list)):
            evidence_list = self.reconstruct_evidence_list(ev)
            turn_record['id'] = i
            self.set_evidence(turn_record, evidence_list)
        return perturbed_record_list

    def restructure_records_strings(self, perturbed_item_list):
        str_list = []
        for i, ev in enumerate(perturbed_item_list):
            evidence_list = self.reconstruct_evidence_list(ev)
            c_str = self.separator.join([self.claim] + evidence_list)
            str_list.append(c_str)
        return np.array(str_list)

    def reconstruct_from_shap(self, item):
        return self.evidence_array[item != -1].tolist()

    def reconstruct_from_lime(self, item):
        evidence_list = item.split(self.separator)
        # remove 'UNKWORDZ' from the evidence list. lime with bow=False replace missing words with 'UNKWORDZ'
        return [ev for ev in evidence_list if ev != 'UNKWORDZ']

    def generate_evidence_list(self, record):
        if xfc.C.TXT_TO_USE in record:
            all_el = record[xfc.C.TXT_TO_USE].split(self.evidence_separator)
            evidence_list = all_el[1:]
            return [ev.strip() for ev in evidence_list]
        elif 'evidence' in record:
            return record['evidence'][0]['content']
        else:
            return record[C.EV_KEY]

    def set_evidence(self, record, evidence_list):
        record[C.TXT_TO_USE] = self.separator.join([self.claim] + evidence_list)
