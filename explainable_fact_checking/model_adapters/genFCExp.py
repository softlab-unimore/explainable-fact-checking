import copy
import json
import os
import tempfile

import numpy as np


class GenFCExp:
    blank_evidence_list = [
        'Here is a blank evidence.',
        'Hello.',
        'In general, most claims of this nature require thorough investigation before conclusions can be drawn.',
        'There is a consensus that such situations should be approached cautiously due to the uncertainty involved.',
        'As is commonly observed, various perspectives can emerge, each contributing to a broader discussion that might not immediately provide clarity.',
        'Numerous factors must be taken into account, making it challenging to draw concrete conclusions based on initial observations alone.',
        'While general principles may apply, it is important to recognize the uniqueness of each scenario and the intricacies involved.',
        'Without further corroboration, it is typically advised to approach such matters with caution, given the potential for misinterpretation or ambiguity.']

    def __init__(self, model_path, random_seed=42, script_name='test_model.sh', nlabels=None):
        self.model_path = model_path
        self.random_seed = random_seed
        self.script_name = script_name
        if nlabels is None:
            nlabels = 3
        self.nlabels = nlabels

    def predict(self, input_list, return_exp=False):
        # add blank evidence to the input list when the evidence is empty
        old_len = len(input_list)
        empty_mask = []
        i = 0
        for record in input_list:
            empty_mask.append(len(record['evidence']) == 0)
            if len(record['evidence']) == 0:
                i += 1
                idx = i % len(GenFCExp.blank_evidence_list)
                record['evidence'] = self.blank_evidence_list[idx:idx + 2]
            record.pop('input_txt_to_use', None)

        if len(input_list) < 50:
            missing = 50 - len(input_list)
        else:
            missing = 10
        # idx_non_empty = empty_mask.index(False) if False in empty_mask else 0  if we want to use the first record as a template
        fill_element = copy.deepcopy(input_list[0])
        to_add = [copy.deepcopy(fill_element) for _ in range(missing)]
        for i, el in enumerate(to_add):
            idx = i % len(GenFCExp.blank_evidence_list)
            el['evidence'] = el['evidence'] + [self.blank_evidence_list[idx]]
        input_list = input_list + to_add  # do not use += here as it will modify the original list
        for i,x in enumerate(input_list):
            x['id'] = i
            nev = len(x['evidence'])
            if len(x['goldtag']) < nev:
                x['goldtag'] = x['goldtag'] + [0] * (nev - len(x['goldtag']))
            else:
                x['goldtag'] = x['goldtag'][:nev]

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_input = os.path.join(tmpdirname, 'tmp.json')
            tmp_out = os.path.join(tmpdirname, 'tmp_out.json')
            with open(tmp_input, 'w') as f:
                json.dump(input_list, f)
            # run the model script
            os.system(
                f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}'
                f' /home/bussotti/XFCresults/datasets_v2/pipelines/Isabelle/{self.script_name}'
                f' {tmp_input} {self.model_path} {tmp_out} {self.nlabels}')
            # read the output file
            with open(tmp_out, 'r') as f:
                out_list = json.load(f)
        out_list = [x[:old_len] for x in out_list]
        input_list = input_list[:old_len]
        for record, empty in zip(input_list, empty_mask):
            if empty:
                record['evidence'] = []

        if return_exp:
            return out_list
        return np.array(out_list[0])

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
