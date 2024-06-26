import copy

import numpy as np
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


class FeverousRecordWrapper:
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

    def __init__(self, record, predictor, separator=r' </s> ', perturbation_mode='only_evidence', explainer='lime',
                 debug=False):
        self.record = record
        self.evidence_index_map = self.generate_evidence_index_map(record)
        self.evidence_array = np.array(self.generate_evidence_list(record))
        self.claim = None
        self.mode = 'normal'
        self.restructure_perturbed_records = self.restructure_perturbed_records_codes
        if 'claim' in record and 'input_txt_to_use' in record:
            assert record['claim'] == record['input_txt_to_use'].split(separator)[
                0], 'Claim and input_txt_to_use do not match'
            self.claim = record['claim']
            self.mode = 'input_txt_to_use'

            if explainer == 'shap':
                self.restructure_perturbed_records = self.restructure_perturbed_records_pos_codes
            elif explainer == 'lime':
                self.restructure_perturbed_records = self.restructure_perturbed_records_strings
            else:
                raise ValueError('Explainer not recognized')
        else:
            assert 'claim' in record, 'Claim not found in record'
        self.separator = separator
        self.predict_method = predictor
        self.id = record['id']
        self.explainer = explainer

        if perturbation_mode == 'only_evidence':
            self.get_text_to_perturb = self.get_evidence_string
        elif perturbation_mode == 'claim_and_evidence':
            # todo adjust with composition wrt the mode of perturbation
            raise (NotImplementedError('Not implemented yet'))

        self.debug = debug
        if debug:
            self.debug_data = {}

    def get_id(self):
        return self.id

    def get_num_evidence(self):
        return len(self.generate_evidence_list(self.record))

    @staticmethod
    def generate_evidence_index_map(record):
        evidence_content = FeverousRecordWrapper.generate_evidence_list(record)
        content_index_map = {}
        for i, ev in enumerate(evidence_content):
            content_index_map[ev] = i
        return content_index_map

    def get_evidence_string(self):
        evidence_content = self.generate_evidence_list(self.record)
        return self.separator.join(evidence_content)

    def get_evidence_list_SHAP(self):
        evidence_list = self.generate_evidence_list(self.record)
        return np.array(evidence_list).reshape(1, -1)

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

    def masker(self, *args):
        return np.ones((1, len(args)))

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

    def restructure_perturbed_records_pos_codes(self, pos_codes):
        perturbed_record_list = [copy.deepcopy(self.record) for _ in range(len(pos_codes))]
        for i, (ev, turn_record) in enumerate(zip(pos_codes, perturbed_record_list)):
            evidence_list = self.evidence_array[ev != -1]
            turn_record['id'] = i
            turn_record['input_txt_to_use'] = self.separator.join([self.claim] + evidence_list.tolist())
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
    def generate_evidence_list(record):
        # if 'input_txt_to_use' in keys of dictionary split the string by '<\e>' the first item is the claim from the second to the end are the evidence
        # remove additional spaces
        # return the list of evidence
        if 'input_txt_to_use' in record:
            all_el = record['input_txt_to_use'].split(FeverousRecordWrapper.evidence_separator)
            evidence_list = all_el[1:]
            # claim = all_el[0]
            return [ev.strip() for ev in evidence_list]
        else:
            return record['evidence'][0]['content']

    @staticmethod
    def set_evidence_content(record, content):
        record['evidence'][0]['content'] = content


