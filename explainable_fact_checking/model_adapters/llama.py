import traceback
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from collections import Counter
import requests
import copy
import argparse
import numpy as np


class LLAMA31Wrapper:
    url = "http://localhost:11434/api/chat"

    type_prompts = {'list0and1':
                        """You are an expert fact-checker. I will provide you a claim and a list of evidence. Basing on them, you should answer in two lines.
                      The first line should contain a list containing n 0s and 1s, n being len(evidence). This list is a noise indicator. For a piece of evidence of index i, 0 means the evidence was not useful to predict the claim label: the piece of evidence is noise, and 0 means that the evidence is useful. [Noisy_EV_i] should then be 0 if evidence i is noisy, and 1 if it is useful. The format of the first line should be 'Gold evidence:[[Noisy_EV_0],[Noisy_EV_1],...,[Noisy_EV_n]]'
                      The second line should contain your predicted label. The label [PREDICTED LABEL] can be $LABELS_TAG$. Answer even if you are not sure. The format of the second line line should be 'Label:[PREDICTED LABEL]'
                      Do not answer anything else than the examples provided.
                      """,

                    # Each evidence associated to a noise label
                    'repeatEachEvWithNOrUTxt':
                        """You are an expert fact-checker. I will provide you a claim and a list of evidence. Basing on them, you should answer in two steps.
                      The first step should be your prediction of useful or noise for each evidence. For each piece of evidence repeat the text and append at the end either ':Noise' or ':Useful'. There should be as many lines as there are evidence. Start this step by the text : 'Evidence with noisetag:'. Then, each line should have the format : '[Evidence_index].[Evidence_text]:[NoiseOrUseful]'
                      The second step should contain your predicted label. The label [PREDICTED LABEL] can be  $LABELS_TAG$. Answer even if you are not sure. The format of the second line step be 'Label:[PREDICTED LABEL]'
                      Do not answer anything else than the examples provided.
                      """,

                    'repeatUsefulEvOnly':
                        """You are an expert fact-checker. I will provide you a claim and a list of evidence. Basing on them, you should answer in two steps.
                      In the first step you should repeat every evidence that are useful to predict a label for the claim. Repeat only those evidence, one evidence piece per line starting with their index, as they are presented to you.  Start this step by the text : 'Useful evidence:'
                      The second step should contain your predicted label. The label [PREDICTED LABEL] can be  $LABELS_TAG$. Answer even if you are not sure. The format of the second step should be 'Label:[PREDICTED LABEL]'
                      """,

                    'listIndexUseful':
                        """You are an expert fact-checker. I will provide you a claim and a list of evidence. Basing on them, you should answer in two lines.
                      The first line should contain a list of useful evidence indexes of each evidence. If an evidence was useful to predict the label, add its index to the list. The format of the first line should be 'Useful evidence index:[[Index_useful_ev1],[Index_useful_ev2],...,[Index_useful_evk]]'
                      The second line should contain your predicted label. The label [PREDICTED LABEL] can be  $LABELS_TAG$. Answer even if you are not sure. The format of the second line line should be 'Label:[PREDICTED LABEL]'
                      """,

                    'noEvidence':
                        """You are an expert fact-checker. I will provide you a claim and a list of evidence. Basing on them, you should answer only with your predicted label [PREDICTED LABEL]. 
                        The label [PREDICTED LABEL] can be  $LABELS_TAG$. Answer even if you are not sure. The format of your output should be $example_out$
                        Do not answer anything else than the examples provided.
                      """,
                    }

    def __init__(self, labels, prompt_type, nb_ex_fs, input_for_fs='', cache_pred_file=None, modelToUse=None,
                 random_seed=42, max_attempts=5, default_label=None):

        if modelToUse is None:
            raise Exception('modelToUse should be provided')
        u = labels.upper().replace('_', ' ').split('.')
        self.id_to_label = {x: u[x] for x in range(len(u))}
        self.label_to_id = {u[x]: x for x in range(len(u))}
        self.labels_tag = '\'' + '\', \''.join(u) + '\''
        self.example_out = ' or '.join([f'\'Label:{l}\'' for l in self.id_to_label.values()])
        if default_label is None:
            default_label = self.id_to_label[0]
            print(f'default_label not provided, setting it to {default_label}')
        default_label = default_label.upper().replace('_', ' ')
        if default_label not in self.id_to_label.values():
            raise Exception(f'default_label {default_label} not in labels {self.id_to_label.values()}')
        self.default_label = default_label
        self.prompt_type = prompt_type
        self.nb_ex_fs = nb_ex_fs
        self.input_for_fs = input_for_fs
        self.max_attempts = max_attempts

        self.cache_pred_file = cache_pred_file
        if cache_pred_file is None or not os.path.exists(cache_pred_file):
            dico = dict()
        else:
            f = open(cache_pred_file, 'r')
            dico = json.load(f)
            f.close()
        # delete cache_pred_file
        # if os.path.exists(cache_pred_file):
        #     os.remove(cache_pred_file)
        # self.dico = {}
        self.dico = dico
        self.modelToUse = modelToUse
        self.random_seed = random_seed

        obj_tr_dico = dict()
        if type(self.input_for_fs) == str and len(self.input_for_fs) > 3:
            f = open(self.input_for_fs)
            obj_tmp = json.load(f)
            f.close()
            obj_tr = []
            for elt in obj_tmp:
                elt['label'] = self.id_to_label[elt['label']]
                obj_tr += [elt]

            for label in list(set([x['label'] for x in obj_tr])):
                obj_tr_dico[label] = []
                for elt in obj_tr:
                    if elt['label'] == label:
                        obj_tr_dico[label] += [elt]

    def get_single_prediction(self, elt, force_rerun=False, tempertaure:float=0):
        prompt_txt = self.build_prompt(elt, self.nb_ex_fs)
        ret_model = self.run_chatGPT(prompt_txt, self.modelToUse, force_rerun=force_rerun, tempertature=tempertaure)
        res_ev = self.convertOutputEvToList(ret_model)
        res_ev = res_ev + [0 for x in range(max(0, len(elt['goldtag']) - len(res_ev)))]
        res_label = self.convertOutputClaimToClaim(ret_model)

        if res_label.upper() not in list(self.label_to_id.keys()):
            raise Exception(f'Label {res_label} not found in {self.label_to_id.keys()}. Full output: {ret_model}')
            # print('checking if label foundable')
            # for key in list(self.label_to_id.keys()):
            #     if key in res_label.upper():
            #         res_label = key
            #         break
        return res_label, res_ev

    def predict(self, input_json):

        # modelToUse="llama3.1:70b"

        obj_tmp = input_json
        obj = []
        for elt in obj_tmp:
            elt['label'] = self.id_to_label[elt['label']]
            obj += [elt]
        obj_dico = dict()
        for label in list(set([x['label'] for x in obj])):
            obj_dico[label] = []
            for elt in obj:
                if elt['label'] == label:
                    obj_dico[label] += [elt]

        prompt_to_use = self.type_prompts[self.prompt_type]

        resbig = dict()
        unsolved_error = 0
        resbig[self.prompt_type] = dict()

        obj_to_use = obj
        # obj_tr_dico=obj_dico

        predlabel = dict()

        predlabelclaim = dict()
        #####INDEP SCORES PER CLAIM LABEL END
        error_output = 0

        for idx, elt in enumerate(obj_to_use):
            try:
                res_label, res_ev = self.get_single_prediction(elt, force_rerun=False)
                # raise Exception('test')
                # res+=[{'claim':elt['claim'],'ret_model':ret_model,'prompt':prompt_txt, 'original_noise':elt['noisetag'], 'original_label':elt['label'],'predicted_noise':res_ev,'predicted_label':res_label,}]
            except Exception as e:
                sflag = False
                for i in range(self.max_attempts - 1):
                    try:
                        res_label, res_ev = self.get_single_prediction(elt, force_rerun=True, tempertaure=0.1 * (i + 1))
                        # raise Exception('test')
                        sflag = True
                        print('Success after', i, 'attempts.')
                        break
                    except Exception as e:
                        pass
                if not sflag:
                    try:
                        res_label, res_ev = self.get_single_prediction(elt, force_rerun=True, tempertaure=0.1 * (i + 1))
                        # raise Exception('test')
                        print('Success after', i+1, 'attempts.')
                    except Exception as e:
                        print(e)
                        # print(e)
                        # traceback.print_exc()
                        unsolved_error += 1
                        res_label, res_ev = self.default_label, [0] * len(elt['goldtag'])
                        print('Unsolved error, setting to default label. Error count:', unsolved_error)


            if len(res_ev) > len(elt['goldtag']):
                error_output += 1
                continue
            label_tmp = self.label_to_id[res_label.upper()]
            predlabel[elt['id']] = [res_ev]
            predlabelclaim[elt['id']] = [label_tmp]
            if len(self.dico) % 100 == 0:
                self.save_dico()
        self.save_dico()
        all_ids = [x['id'] for x in obj_to_use]
        predictions_fin_claims = []
        predictions_fin_evs = []
        for elt in all_ids:
            if elt in predlabel.keys():
                predictions_fin_claims += predlabelclaim[elt]
                predictions_fin_evs += predlabel[elt]

            else:
                predictions_fin_claims += [[]]
                predictions_fin_evs += [[]]

        n_pred = len(predictions_fin_claims)
        pred_ohe = np.zeros((n_pred, len(self.id_to_label)))
        pred_ohe[np.arange(n_pred), predictions_fin_claims] = 1
        if not self.prompt_type == 'noEvidence':
            return pred_ohe, predictions_fin_evs
        else:
            return pred_ohe

    def save_dico(self):
        if self.cache_pred_file is not None:
            with open(self.cache_pred_file, 'w') as f:
                json.dump(self.dico, f)

    def run_chatGPT(self, prompt, modelToUse, force_rerun=False, tempertature:float=0):
        if force_rerun is False:
            if prompt + modelToUse in self.dico:
                return self.dico[prompt + modelToUse]

        data = {
            "model": modelToUse,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": tempertature
            },
        }

        # Send the POST request
        response = requests.post(self.url, json=data)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            result = response.json()
        # print(result)
        txt = result['message']['content']
        self.dico[prompt + modelToUse] = txt
        return txt

    @staticmethod
    def fctToFilterMulti(x):
        tmp = [y for y in x.split('\n') if len(y.strip()) > 5][-1].lower().replace('verdict:', '')
        if not tmp in ['cannot say', 'misinformation', 'not misinformation']:
            for elt in ['cannot say', 'not misinformation', 'misinformation']:
                if elt in tmp:
                    return elt
        for elt in x.split('\n')[::-1]:
            tmp = elt.lower()
            for elt2 in ['cannot say', 'not misinformation', 'misinformation']:
                if elt2 in tmp:
                    return elt2
        return tmp

    def format_example(elt, type_prompt, goldLabel=True):
        returntxt = 'Claim:' + elt['claim']

        returntxt += 'Evidence:\n' + '\n'.join(
            [str(ix) + '.' + x.replace('\n', '') for ix, x in enumerate(elt['evidence'])])
        if goldLabel:
            returntxt += '\nExpected output:\n'
            if type_prompt == 'list0and1':
                returntxt += 'Gold evidence:' + str(elt['evidence'])
            if type_prompt == 'repeatEachEvWithNOrUTxt':
                returntxt += 'Evidence with goldtag:\n' + '\n'.join(
                    [str(ix) + '.' + x.replace('\n', '') + ' : ' + ('Noise' if y == 0 else 'Useful') for ix, (x, y) in
                     enumerate(zip(elt['evidence'], elt['goldtag']))])
            if type_prompt == 'repeatUsefulEvOnly':
                returntxt += 'Useful evidence:\n' + '\n'.join([str(ix) + '.' + x.replace('\n', '') for ix, (x, y) in
                                                               enumerate(zip(elt['evidence'], elt['goldtag'])) if
                                                               y == 1])
            if type_prompt == 'listIndexUseful':
                returntxt += 'Useful evidence index:\n' + str(
                    [ix for ix, (x, y) in enumerate(zip(elt['evidence'], elt['goldtag'])) if y == 1])
            if type_prompt == 'noEvidence':
                returntxt += ''
            returntxt += '\nLabel:' + elt['label']

        returntxt += '\n'
        return returntxt

    def build_prompt(self, elt2, nb_ex_fs):
        base_prompt = self.type_prompts[self.prompt_type].replace('$LABELS_TAG$', self.labels_tag)
        base_prompt = base_prompt.replace('$example_out$', self.example_out)
        if nb_ex_fs > 0:
            base_prompt += '\nHere are some examples with your expected output\n'
            keys_labels_tr = [x for x in self.obj_to_use_tr.keys()]
            for ielt in range(nb_ex_fs):
                elt = self.obj_to_use_tr[keys_labels_tr[ielt % len(keys_labels_tr)]][ielt // len(keys_labels_tr)]
                base_prompt += '\nExample ' + str(ielt) + ':\n'
                base_prompt += self.format_example(elt, self.prompt_type, goldLabel=True)

        base_prompt += '\nThe claim and evidence you should use are:\nClaim: ' + elt2['claim']
        ev_list = elt2['evidence']
        if len(ev_list) == 0:
            ev_list = ['No evidence provided']
        base_prompt += ' Evidence:\n' + '\n'.join(
            [str(ix) + '.' + x.replace('\n', '') for ix, x in enumerate(ev_list)])

        return base_prompt

    def convertOutputEvToList(self, output):
        try:
            if self.prompt_type == 'list0and1':
                output = output.replace('[', '').replace(']', '')
                output = [x for x in output.split('\n') if len(x) > 5]
                list_nb = [int(u) for u in output[0].split('Gold evidence:')[1].split(',')]
                return list_nb
            elif self.prompt_type == 'repeatEachEvWithNOrUTxt':
                output = [x for x in output.split('\n') if len(x) > 5]
                list_index_useful = []
                for line in output[1:-1]:
                    if line.split(':')[-1].strip().lower() == 'useful':
                        list_index_useful += [int(line.split('.')[0])]
                max_liu = max(list_index_useful) if len(list_index_useful) else 0
                return [1 if x not in list_index_useful else 0 for x in range(max_liu + 1)]
            elif self.prompt_type == 'repeatUsefulEvOnly':
                output = [x for x in output.split('\n') if len(x) > 5]
                list_index_useful = [int(x.split('.')[0]) for x in output[1:-1]]
                return [1 if x not in list_index_useful else 0 for x in range(max(list_index_useful) + 1)] if len(
                    list_index_useful) > 0 else []

            elif self.prompt_type == 'listIndexUseful':
                output = output.replace('[', '').replace(']', '')
                output = [x for x in output.split('\n') if len(x) > 5]
                tmp = output[0].split('Useful evidence index:')[1]
                list_nb = [int(u) for u in tmp.split(',')] if len(tmp.replace('[', '').replace(']', '')) > 0 else []
                return [1 if x not in list_nb else 0 for x in range(max(list_nb) + 1)] if len(list_nb) > 0 else []
            elif self.prompt_type == 'noEvidence':
                return []
        except Exception as e:
            print(e)
            print(output)
            raise e

    def convertOutputClaimToClaim(self, output):
        output = [x for x in output.split('\n') if len(x) > 5]
        return output[-1].split('Label:')[1].strip().replace('*', '').replace('[', '').replace(']', '')

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input and output file names.")
    parser.add_argument("--input", help="The name of the input file.")
    parser.add_argument("--input_for_fs", help="The name of the input file.")
    parser.add_argument("--input_model", help="The name of the input model .")
    parser.add_argument("--prompt_type", help="The name of the prompt to use.")
    parser.add_argument("--output", help="The name of the output file.")
    parser.add_argument("--labels",
                        help="Map index to Existing labels txt, separated by dots and spaces replaced by _. Ex : SUPPORTS.REFUTES.NOT_ENOUGH_INFO.")
    parser.add_argument("--nb_ex_fs", help="Number of few shots examples.")
    args = parser.parse_args()

    # Pass the arguments to the main function
    # main(args.input, args.input_model, args.output, args.labels, args.prompt_type, int(args.nb_ex_fs),
    #      args.input_for_fs)
