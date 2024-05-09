import json
import os
import time

import numpy as np

import sys
sys.path.append('/homes/bussotti/feverous_work/feverousdata')
sys.path.append('/homes/bussotti/feverous_work/feverousdata/feverous')


import predictor_universal


def run_exp_multitest(exp_folder, dest_folder, train_db, train_dev_file, dev_db, dev_testfile, suffix,
                      pretrained_retriever_to_use, pretrained_predictor_to_use, usinggt, usingretriever):
    base_path = '/homes/bussotti/feverous_work/feverousdata'
    list_exp_names = os.listdir('//homes/bussotti/feverous_work/feverousdata/' + exp_folder + '/')
    list_exp_names2 = [x for x in list_exp_names if '.jsonl' in x]
    list_files_to_add = ['//homes/bussotti/feverous_work/feverousdata/' + exp_folder + '/' + x for x in list_exp_names2]
    list_exp_names = []

    for i in range(len(dest_folder)):
        list_exp_names += [[dest_folder[i] + '/' + x.split('.')[0] for x in list_exp_names2]]
        if not os.path.exists(base_path + '/' + dest_folder[i]):
            os.mkdir(base_path + '/' + dest_folder[i])

    for ft in enumerate(list_files_to_add):

        db_path = base_path + '/' + train_db

        train_devset = train_dev_file

        data_path = base_path
        f2 = open(ft[1], 'r')
        f3 = open(base_path + "/train.jsonl", 'w')
        print(ft[1])
        for line in f2:
            f3.write(line)
        f3.close()
        f_params = open(ft[1].split('.json')[0] + '_params.json', 'r')
        obj_params = json.load(f_params)
        f_params.close()

        for i in range(len(dest_folder)):
            exp_name = list_exp_names[i][ft[0]]
            dev_db_path = base_path + '/' + dev_db[i]
            test_devset = dev_testfile[i]

            if os.path.exists(data_path + "/" + exp_name + '_predictor_scores.json'):
                print("|i> Predictor " + ft[1] + " for " + dest_folder[i] + " already done .")
                continue
            start = time.time()
            predictor_universal.exec_predictor_alt_devdb_quick(db_path, data_path, exp_name, dev_db_path, train_devset,
                                                               test_devset, False, usinggt=usinggt,
                                                               usingretriever=usingretriever)

            end = time.time()
            obj_params['execution_time'] = end - start
            f_params = open(data_path + "/" + exp_name + '_predictor_params.json', 'w')
            json.dump(obj_params, f_params)
            f_params.close()
        else:
            print("|i> " + ft[1] + " already done .")


def run_exp_multitest_wsavedmodel(dest_folder, train_db, train_dev_file, dev_db, dev_testfile, suffix,
                                  pretrained_retriever_to_use, pretrained_predictor_to_use, usinggt, usingretriever,
                                  pathModel):
    base_path = '/homes/bussotti/feverous_work/feverousdata'

    for i in range(len(dest_folder)):
        if not os.path.exists(base_path + '/' + dest_folder[i]):
            os.mkdir(base_path + '/' + dest_folder[i])
        db_path = base_path + '/' + train_db

        train_devset = train_dev_file

        data_path = base_path
        exp_name = dest_folder[i]
        dev_db_path = base_path + '/' + dev_db[i]
        test_devset = dev_testfile[i]

        # if os.path.exists(data_path + "/" + exp_name + '_predictor_scores.json'):
        #     print("|i> Predictor for " + dest_folder[i] + " already done .")
        #     continue
        start = time.time()
        predictor_universal.exec_predictor_alt_devdb_quick(db_path, data_path, exp_name, dev_db_path, train_devset,
                                                           test_devset, True, usinggt=usinggt,
                                                           usingretriever=usingretriever, pathModel=pathModel)

        end = time.time()


    print("Prediction finished.")


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
        self.base_path = '/homes/bussotti/feverous_work/feverousdata/'
        self.tmp_file = 'AB/tmp/tmp_records.jsonl'
        self.tmp_pred_file = 'AB/tmp/tmp_predictions'

    def predict(self, records):
        records = [records[0]] + records

        with open(self.base_path + self.tmp_file, 'w') as file:
            for record in records:
                if record['label'] == 'NOT ENOUGH INFO':
                    record['label'] = 'REFUTES'
                file.write(json.dumps(record) + '\n')

        res = run_exp_multitest_wsavedmodel([self.tmp_pred_file], 'jf_home/feverous_wikiv1.db',
                                            'feverous_dev_challenges_sentencesandtable.jsonl',
                                            ['jf_home/feverous_wikiv1.db'],
                                            [self.tmp_file], '', None, None, True, True, 'models_fromjf270623or')

        # Prediciton format example
        # {   "claim": "Paul Dicks (born in 1950) was a Speaker of the Newfoundland and Labrador House of Assembly, preceding Thomas Lush and succeeded by Lloyd Snow.",
        #     "label": "SUPPORTS", "predicted_label": "SUPPORTS", "label_match": true}

        with open(self.base_path + self.tmp_pred_file + '_predictor.jsonl', 'r') as file:
            predictions = [json.loads(line) for line in file if line != '\n']
        self.predictions = predictions
        return np.array([pred['predicted_scores'] for pred in predictions])
