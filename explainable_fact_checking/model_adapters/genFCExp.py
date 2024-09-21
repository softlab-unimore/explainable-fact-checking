import copy
import json
import os
import tempfile

import numpy as np


class GenFCExp:
    blank_evidence_list = [
        ' ', '.',
        'In general, most claims of this nature require thorough investigation before conclusions can be drawn.',
        'There is a consensus that such situations should be approached cautiously due to the uncertainty involved.',
        'As is commonly observed, various perspectives can emerge, each contributing to a broader discussion that might not immediately provide clarity.',
        'Numerous factors must be taken into account, making it challenging to draw concrete conclusions based on initial observations alone.',
        'While general principles may apply, it is important to recognize the uniqueness of each scenario and the intricacies involved.',
        'Without further corroboration, it is typically advised to approach such matters with caution, given the potential for misinterpretation or ambiguity.']

    def __init__(self, model_path, random_seed=42):
        self.model_path = model_path
        self.random_seed = random_seed

    def predict(self, input_list, return_exp=False):
        # add blank evidence to the input list when the evidence is empty
        old_evidence = []
        i = 0
        for record in input_list:
            old_evidence.append(copy.deepcopy(record['evidence']))
            if len(record['evidence']) == 0:
                # record['evidence'] = self.blank_evidence_list[i % len(self.blank_evidence_list)]
                i += 1
                record['evidence'] = [' . ']
        input_list += input_list[-2:]

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_input = os.path.join(tmpdirname, 'tmp.json')
            tmp_out = os.path.join(tmpdirname, 'tmp_out.json')
            with open(tmp_input, 'w') as f:
                json.dump(input_list, f)
            # run the model script
            os.system(
                f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")} /home/bussotti/XFCresults/datasets_v2/pipelines/Isabelle/test_model.sh {tmp_input} {self.model_path} {tmp_out}')
            # read the output file
            with open(tmp_out, 'r') as f:
                out_list = json.load(f)
        input_list = input_list[:-2]
        for record, ev in zip(input_list, old_evidence):
            record['evidence'] = ev
        if return_exp:
            return out_list
        return np.array(out_list[0][:48])

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
