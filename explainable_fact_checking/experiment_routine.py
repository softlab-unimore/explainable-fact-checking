import sys

import numpy as np

from explainable_fact_checking import explain_with_lime, FeverousModelAdapter
from feverous_model import run_exp_multitest_wsavedmodel

sys.path.insert(0, "/homes/bussotti/feverous_work/feverousdata/feverous/")
# sys.path.insert(0, "/homes/bussotti/feverous_work/feverousdata/")

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json

#Root directory
base_path='/homes/bussotti/feverous_work/feverousdata'

# main thing to start the experiment
if __name__ == "__main__":
    # fake predictor function that takes in input a set of list and returns a list of random predictions with shape (len(strings), 1)
    def fake_predictor(restructured_records):
        # increasing prediction with the length of the evidence
        predictions = np.array([len(x['evidence'][0]['content']) for x in restructured_records]).reshape(-1, 1)
        # scale the predictions between 0 and 1
        predictions = predictions / np.max(predictions)
        return predictions


    fc_model = FeverousModelAdapter()
    predictor = fc_model.predict
    for name in ['ex_AB_00.jsonl', 'feverous_dev_ST_01.jsonl', 'feverous_dev_SO_01.jsonl']:
        input_file_path = os.path.join('/homes/bussotti/feverous_work/feverousdata/AB/', name)

        explain_with_lime(input_file_path,
            #'sub_stcetb.json',
                          predictor=predictor, top=100)
