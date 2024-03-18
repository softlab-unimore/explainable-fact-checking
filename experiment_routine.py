
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cell_retriever
import predictor_universal
import json
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import torch
import shutil
import traceback

#Root directory
base_path='/homes/bussotti/feverous_work/feverousdata'


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

        if os.path.exists(data_path + "/" + exp_name + '_predictor_scores.json'):
            print("|i> Predictor " + ft[1] + " for " + dest_folder[i] + " already done .")
            continue
        start = time.time()
        predictor_universal.exec_predictor_alt_devdb_quick(db_path, data_path, exp_name, dev_db_path, train_devset,
                                                           test_devset, True, usinggt=usinggt,
                                                           usingretriever=usingretriever, pathModel=pathModel)

        end = time.time()

    else:
        print("|i> " + ft[1] + " already done .")



AB_path = '/homes/bussotti/feverous_work/feverousdata/AB/'
example_file = AB_path + 'sub_stcetb.json'


# main thing to start the experiment
if __name__ == "__main__":
    with open(example_file, 'r') as file:
        data = [json.loads(line) for line in file if line != '\n']

    YOURTESTFILE = 'feverous_dev_challenges_sentencesandtable.jsonl'
    # Just change the YOURTESTFILE with the name of your test file in .jsonl. It should be in the folder 'feverousdata'
    # The results are put in the folder res_exp_andrea. It contains a jsonl with the results. For each example you can see the predicted label
    # !!! After each run think to empty the folder res_exp_andrea, otherwise it will skip the next computation as it will consider that it was already done

    run_exp_multitest_wsavedmodel(['AB/res_exp_andrea'], 'jf_home/feverous_wikiv1.db',
                                  'feverous_dev_challenges_sentencesandtable.jsonl', ['jf_home/feverous_wikiv1.db'],
                                  [YOURTESTFILE], '', None, None, True, True, 'models_fromjf270623or')


