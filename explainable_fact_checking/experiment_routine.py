import sys

import numpy as np

from explainable_fact_checking import explain_with_lime, FeverousModelAdapter
import explainable_fact_checking as xfc
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Root directory
base_path = '/homes/bussotti/feverous_work/feverousdata'

# main thing to start the experiment
if __name__ == "__main__":
    # fake predictor function that takes in input a set of list and returns a list of
    # random predictions with shape (len(strings), 1)
    def fake_predictor(restructured_records):
        # increasing prediction with the length of the evidence
        # predictions = np.array([len(x['evidence'][0]['content']) for x in restructured_records]).reshape(-1, 1)
        # predictions = predictions / np.max(predictions)

        predictions = np.random.rand(len(restructured_records), 1)
        # scale the predictions between 0 and 1
        predictions = np.concatenate([predictions, 1-predictions, np.zeros_like(predictions)], axis=1)
        return predictions


    fc_model = FeverousModelAdapter()
    predictor = fc_model.predict
    for name in [
        'original_TO_01_formatted.jsonl',
        # 'ex_AB_00.jsonl',
        # 'feverous_dev_ST_01.jsonl',
        # 'feverous_dev_SO_01.jsonl',
    ]:
        input_file_path = os.path.join('/homes/bussotti/feverous_work/feverousdata/AB/', name)

        xfc.wrappers.explain_with_SHAP(input_file_path, predictor=predictor,
                                       output_dir='/homes/bussotti/feverous_work/feverousdata/AB/SHAP_explanations',
                                       num_samples=500,
                                       top=1000)

        # xfc.wrappers.explain_with_lime(input_file_path, predictor=predictor,
        #                   output_dir='/homes/bussotti/feverous_work/feverousdata/AB/lime_explanations', num_samples=500,
        #                   top=1000)
