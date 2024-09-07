import sys

from feverous_model import run_exp_multitest_wsavedmodel

sys.path.insert(0, "/homes/bussotti/feverous_work/feverousdata/feverous/")

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json

#Root directory
base_path='/homes/bussotti/feverous_work/feverousdata'

def test_run_exp_multitest_wsavedmodel():
    AB_path = '/homes/bussotti/feverous_work/feverousdata/AB/'
    example_file = AB_path + 'sub_stcetb.json'

    with open(example_file, 'r') as file:
        data = [json.loads(line) for line in file if line != '\n']

    res = run_exp_multitest_wsavedmodel(['AB/res_exp_andrea'], 'jf_home/feverous_wikiv1.db',
                                        'feverous_dev_challenges_sentencesandtable.jsonl',
                                        ['jf_home/feverous_wikiv1.db'],
                                        ['AB/sub_stcetb.json'], '', None, None, True, True, 'models_fromjf270623or')


