import copy
import json
import os

import nltk
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from explainable_fact_checking.datasets_loaders import RobertaTokenizerWrapper, load_std_dataset

from explainable_fact_checking import C
import explainable_fact_checking as xfc


class AveritecPre:

    @staticmethod
    def get_noisy_evidence(item, gold_df, flat_noise_filtered, flat_noise, max_length, random_seed=42):
        id = item.get('id', item.get('claim_id', None))
        if id is None:
            raise ValueError('Claim id not found')

        t_dict = item
        budget = max_length - t_dict['ntokens']
        ev_to_add = []
        eg = gold_df.query(f'claim_id == {id}')
        gold_answer_list = eg['full_text']
        if not flat_noise_filtered.empty:
            retrieved = flat_noise_filtered.query(f'claim_id == {id}').copy()

            # Add first retrieved evidence
            candidate_answer_list = retrieved['full_text']
            sim_score = AveritecPre.calculate_bm25_score(queries=gold_answer_list, candidates=candidate_answer_list).sum(0)
            retrieved['sim_score'] = sim_score


            # select retrieved with ntokens < budget
            pool_retrieved = retrieved[retrieved['ntokens'] <= budget]
            if len(pool_retrieved) == 0:
                return pd.DataFrame(), budget

            # select ones with lowest similarity score
            least_similar = pool_retrieved[pool_retrieved['sim_score'] == pool_retrieved['sim_score'].min()]

            # sort by token length
            least_similar = least_similar.sort_values('ntokens')

            to_add = least_similar.iloc[0]
            budget -= to_add['ntokens']
            ev_to_add.append(to_add.copy())

            # debug
            # noise_ev = AveritecPre.convert_ev_df2averitec(pd.DataFrame(ev_to_add))
            # k =copy.deepcopy(item)
            # k['questions'] = k['questions'] + noise_ev
            # k = AveritecPre.preprocess_AVERITEC([k], prepend_question=True)[0]
            # assert tokenizer_h.ntokens(tokenizer_h.strucutre(k['claim'], k['evidence'])) <= 512
            # assert tokenizer_h.ntokens(tokenizer_h.strucutre(k['claim'], k['evidence'])) - tokenizer_h.ntokens(tokenizer_h.strucutre(k['claim'], k['evidence'][:-1])) == to_add['ntokens']
            # tokenizer_h.tokenizer.convert_ids_to_tokens(tokenizer_h.tokenizer(tokenizer_h.strucutre(k['claim'], k['evidence']))['input_ids'])[:41]

            # remove added evidence from pool retrieved
            pool_retrieved = pool_retrieved.drop(to_add.name)

        # Add single unrelated evidence
        # from flat_noisy_qa select evidence with different claim_id
        # select items fitting budget and with lowest counter (added less times)
        # compute similarity between a sample of 100 and gold evidence plus claim as query
        # select one with lowest similarity score, with lowest token length when equal score
        # update counter and budget

        m1 = flat_noise['claim_id'] != id
        m2 = flat_noise['ntokens'] <= budget

        pool_unrelated = flat_noise[m1 & m2]
        if len(pool_unrelated) == 0:
            return pd.DataFrame(ev_to_add), budget
        mcounter = pool_unrelated['counter'] == pool_unrelated['counter'].min()
        pool_unrelated = pool_unrelated[mcounter]

        if len(pool_unrelated) >= 100:
            tdf = pool_unrelated.sample(100, random_state=random_seed)
        else:
            tdf = pool_unrelated

        sim_score = AveritecPre.calculate_bm25_score(queries=gold_answer_list + [t_dict['claim']],
                                                     candidates=tdf['full_text']).sum(0)
        tdf['sim_score'] = sim_score
        if flat_noise_filtered.empty:
            pool_retrieved = tdf

        tdf = tdf[sim_score == sim_score.min()]

        to_add = tdf[tdf['ntokens'] == tdf['ntokens'].min()].iloc[0]
        budget -= to_add['ntokens']
        to_add['counter'] += 1
        flat_noise.update(to_add.to_frame().T)
        ev_to_add.append(to_add.copy())

        # debug
        # noise_ev = AveritecPre.convert_ev_df2averitec(pd.DataFrame(ev_to_add))
        # k = copy.deepcopy(item)
        # k['questions'] = k['questions'] + noise_ev
        # k = AveritecPre.preprocess_AVERITEC([k], prepend_question=True)[0]
        # assert tokenizer_h.ntokens(tokenizer_h.strucutre(k['claim'], k['evidence'])) <= 512
        # assert tokenizer_h.ntokens(tokenizer_h.strucutre(k['claim'], k['evidence'])) - tokenizer_h.ntokens(
        #     tokenizer_h.strucutre(k['claim'], k['evidence'][:-1])) == to_add['ntokens']
        # tokenizer_h.tokenizer.convert_ids_to_tokens(tokenizer_h.tokenizer(tokenizer_h.strucutre(k['claim'], k['evidence']))['input_ids'])[:41]

        while True:
            pool_retrieved = pool_retrieved[pool_retrieved['ntokens'] <= budget]
            if len(pool_retrieved) == 0:
                break
            tpool = pool_retrieved[pool_retrieved['sim_score'] == pool_retrieved['sim_score'].min()]
            tpool = tpool.sort_values('ntokens')
            to_add = tpool.iloc[0]
            budget -= to_add['ntokens']
            ev_to_add.append(to_add.copy())

            # debug
            # noise_ev = AveritecPre.convert_ev_df2averitec(pd.DataFrame(ev_to_add))
            # k = copy.deepcopy(item)
            # k['questions'] = k['questions'] + noise_ev
            # k = AveritecPre.preprocess_AVERITEC([k], prepend_question=True)[0]
            # assert tokenizer_h.ntokens(tokenizer_h.strucutre(k['claim'], k['evidence'])) <= 512
            # assert tokenizer_h.ntokens(tokenizer_h.strucutre(k['claim'], k['evidence'])) - tokenizer_h.ntokens(
            #     tokenizer_h.strucutre(k['claim'], k['evidence'][:-1])) == to_add['ntokens']
            # tokenizer_h.tokenizer.convert_ids_to_tokens(tokenizer_h.tokenizer(tokenizer_h.strucutre(k['claim'], k['evidence']))['input_ids'])[:41]

            # remove added evidence from pool retrieved
            pool_retrieved = pool_retrieved.drop(to_add.name)
            if len(ev_to_add) > 5:
                break
        return pd.DataFrame(ev_to_add), budget

    @staticmethod
    def calculate_bm25_score(queries: list[str], candidates):
        score_list = []
        queries = [nltk.word_tokenize(r) for r in queries]
        candidates = [nltk.word_tokenize(c) for c in candidates]
        for reference in queries:
            prompt_bm25 = BM25Okapi(candidates)
            scores = prompt_bm25.get_scores(reference)
            score_list.append(scores)
        return np.array(score_list)

    @staticmethod
    def convert_ev_df2averitec(evidence_df: pd.DataFrame):
        evidence_list = []
        for i, el in evidence_df.iterrows():
            evidence_list.append(dict(question=el['question'],
                                      answers=[
                                          dict(answer=el['answer'], source_url=el['source_url'],
                                               answer_type=el['answer_type'])
                                      ]))

        return evidence_list

    @staticmethod
    def filter_noisy_same_url(data_dict_list, qa_noisy):
        """
        Filter noisy evidence with same source_url of gold evidence

        Parameters
        ----------
        data_dict_list : list
            List of dictionaries with the dataset
        qa_noisy : list
            List of dictionaries with the noisy evidence

        Returns
        -------
        list
            List of dictionaries with the noisy evidence filtered.

        """
        processed_dataset = copy.deepcopy(data_dict_list)
        qa_noisy_filter_1 = []
        for i, (gold, retrieved) in enumerate(zip(processed_dataset, qa_noisy)):
            if gold['claim'] != retrieved['claim']:
                print(f'ERROR in claim: {gold["claim"]} != {retrieved["claim"]} at index {i}')
                break
            tmp_qa = copy.deepcopy(retrieved)
            # gold['gold_evidence'] = [1] * len(list(gold['questions']))
            gold_source_urls = [a['source_url'] for q in gold['questions'] for a in q['answers']]
            # gold['same_site'] = []
            for q in retrieved['questions']:
                for a in q['answers']:
                    if a['source_url'] not in gold_source_urls:
                        pass
                        # gold['questions'].append(q)
                        # gold['gold_evidence'].append(0)
                    else:
                        # gold['same_site'].append(q)
                        # remove i-th question
                        tmp_qa['questions'].remove(q)
            qa_noisy_filter_1.append(tmp_qa)
        return qa_noisy_filter_1

    @staticmethod
    def prepare_qa(qa):
        qa_flat = []
        for i, record in tqdm(enumerate(qa)):
            claim_id = record.get('claim_id', record.get('id', i))
            for qidx, q in enumerate(record['questions']):
                for aidx, a in enumerate(q['answers']):
                    qa_flat.append(
                        dict(claim_id=claim_id, claim=record['claim'],
                             question=AveritecPre.adjest_question_txt(q['question']), question_id=qidx,
                             answer_id=aidx, **a))
        res_df = pd.DataFrame(qa_flat)
        res_df['full_text'] = res_df.apply(AveritecPre.qa2txt, axis=1)
        res_df['ntokens'] = res_df['full_text'].apply(tokenizer_h.ntokens)
        return res_df

    @staticmethod
    def qa2txt(qa_row, prepend_question=True):
        """
        Function to convert question and answers to a single text. This is to be used in pd.DataFrame.apply

        Parameters
        ----------
        qa_row : pd.Series
            Series with the question and answers
        prepend_question : bool
            If True, prepend the question to the evidence text

        Returns
        -------
        str
            Text with the question and answers

        """
        t_txt = ''
        if prepend_question:
            t_txt = AveritecPre.adjest_question_txt(qa_row['question'])
            t_txt += ' '
        t_txt += qa_row['answer']
        if qa_row['answer_type'] == 'Boolean' and 'boolean_explanation' in qa_row:
            t_txt += '. ' + qa_row['boolean_explanation']
        t_txt = t_txt.replace('\n', ' ')
        t_txt = t_txt.replace('  ', ' ')
        t_txt = ' '.join(t_txt.split())
        return t_txt

    @staticmethod
    def preprocess_AVERITEC(dataset, prepend_question=True):
        """
        Preprocess AVERITEC dataset by creating a list of evidence texts for each evidence in the evidence_txt_list field
        and adding an id field to each evidence.

        Parameters
        ----------
        dataset : list
            List of dictionaries with the dataset
        prepend_question : bool
            If True, prepend the question to the evidence text

        Returns
        -------
        list
            List of dictionaries with the dataset with the evidence text list

        """
        for i, e in enumerate(dataset):
            if 'id' not in e:
                e['id'] = i
            e[C.EV_KEY] = []
            if 'questions' in e:
                for q in e['questions']:
                    t_txt = AveritecPre.convert_qa_dict_2txt(q, prepend_question)
                    e[C.EV_KEY].append(t_txt)

            if 'goldtag' not in e:
                e['goldtag'] = [1] * len(e[C.EV_KEY])
        return dataset

    @staticmethod
    def convert_qa_dict_2txt(qa_dict, prepend_question=True):
        """
        Convert a dictionary with question and answers to a text with the question and answers.

        Parameters
        ----------
        qa_dict : dict
            Dictionary with questions and answers with the following structure:
            {
                'question': str,
                'answers': [
                    {
                        'answer': str,
                        'answer_type': str,
                        'boolean_explanation': str
                    }
                ]
            }
        prepend_question : bool
            If True, prepend the question to the evidence text

        Returns
        -------
        str
            Text with the question and answers

        """
        t_txt = ''
        if prepend_question:
            t_txt = AveritecPre.adjest_question_txt(qa_dict['question'])
            t_txt += ' '
        for a in qa_dict['answers']:
            t_txt += a['answer']
            if a['answer_type'] == 'Boolean' and 'boolean_explanation' in a:
                t_txt += '. ' + a['boolean_explanation']
        t_txt = t_txt.replace('\n', ' ')
        t_txt = t_txt.replace('  ', ' ')
        t_txt = ' '.join(t_txt.split())
        return t_txt

    @staticmethod
    def adjest_question_txt(question):
        return question + '?' if not question.endswith('?') else question


if __name__ == '__main__':
    '''
    Gather all retreived evidence QA form with id of their claim.

    Find actual token length of claim and evidence, and the remaining tokens for noise.
    
    Find candidate noisy evidence from retrieved evidence.
    
    Filter retrieved evidence with same source_url of gold evidence.
    
    Sort by BM25 similarity score, with equal score, sort by token length.
    '''

    random_seed = 42
    rnd = np.random.RandomState(random_seed)
    # reload datasets

    ds_path = os.path.join(xfc.experiment_definitions.E.DATASET_DIR_V2, 'AVERITEC')
    preprocessed_path = os.path.join(ds_path, 'preprocessed')
    os.makedirs(preprocessed_path, exist_ok=True)
    files = ['train', 'dev', 'test']
    datasets = {}
    for f in files:
        t_name = f
        with open(os.path.join(preprocessed_path, t_name + '.json'), 'r') as f_in:
            datasets[f] = json.load(f_in)

    qa_noisy_path = hf_hub_download(repo_id="chenxwh/AVeriTeC", filename='data_store/dev_top_k_qa.json',
                                    cache_dir=ds_path,
                                    revision='7f83bddcd8f16570446c51578ab3ddf6a9d350b1')
    qa_items = []
    with open(qa_noisy_path) as f:
        for line in f:
            qa_items.append(json.loads(line))

    qa_noisy = []
    for item in qa_items:
        tmp_qa = []
        for qa in item['bm25_qau']:
            tmp_qa.append(
                dict(question=qa[0], answers=[dict(answer=qa[1], source_url=qa[2], answer_type='Extractive')]))
        tmp_record = copy.deepcopy(item)
        tmp_record.pop('bm25_qau')
        tmp_record['questions'] = tmp_qa
        qa_noisy.append(tmp_record)

    tokenizer_h = RobertaTokenizerWrapper()

    phase='train'
    train_preprocessed = AveritecPre.preprocess_AVERITEC(datasets[phase], prepend_question=True)
    converted_path = os.path.join(xfc.experiment_definitions.E.DATASET_DIR_V3, 'AVERITEC', 'converted')
    os.makedirs(converted_path, exist_ok=True)

    with open(os.path.join(converted_path, 'train_no_noise.json'), 'w') as f_out:
        json.dump(train_preprocessed, f_out)

    gold_df = AveritecPre.prepare_qa(datasets[phase])
    flat_noise = AveritecPre.prepare_qa(qa_noisy)
    flat_noise['counter'] = 0

    for el in datasets[phase]:
        el['ntokens'] = tokenizer_h.ntokens(
            tokenizer_h.strucutre(el['claim'], el['evidence']))
    for el in tqdm(datasets[phase]):
        ev_to_add, b = AveritecPre.get_noisy_evidence(el, gold_df, pd.DataFrame(), flat_noise,
                                                      tokenizer_h.max_length,
                                                      random_seed=random_seed)
        gt_ml = el['ntokens'] > 512
        assert b >= 0 or el['ntokens'] > 512, f'Negative budget {b} for claim {el["claim"]}'
        noise_ev = AveritecPre.convert_ev_df2averitec(ev_to_add)
        el['questions'] = el['questions'] + noise_ev
        el['goldtag'] += [0] * len(noise_ev)
        # shuffle randomly questions and goldtag according to the new evidence
        shuffled_idx = rnd.permutation(len(el['questions']))
        el['questions'] = [el['questions'][i] for i in shuffled_idx]
        el['goldtag'] = [el['goldtag'][i] for i in shuffled_idx]

        #debug
        # el1 = AveritecPre.preprocess_AVERITEC([el], prepend_question=True)[0]
        # el1['ntokens'] = tokenizer_h.ntokens(tokenizer_h.strucutre(el1['claim'], el1['evidence']))
        # assert (el1['ntokens'] > 512) == gt_ml, f'Claim {el1["claim"]} has {el1["ntokens"]} tokens'

    _ = AveritecPre.preprocess_AVERITEC(datasets[phase], prepend_question=True)
    for el in datasets[phase]:
        el['ntokens'] = tokenizer_h.ntokens(
            tokenizer_h.strucutre(el['claim'], el['evidence']))
    print(pd.Series([len(e['evidence']) for e in datasets[phase]]).value_counts())
    # for el in datasets[phase]:
    #     el['ntokens'] = tokenizer_h.ntokens(tokenizer_h.strucutre(el['claim'], el['evidence']))
    print(pd.Series([el['ntokens'] < 512 for el in datasets[phase]]).value_counts())


    with open(os.path.join(converted_path, 'train.json'), 'w') as f_out:
        json.dump(datasets[phase], f_out)

    dev_preprocessed = load_std_dataset(converted_path, 'train.json')
    print(len(dev_preprocessed))
    print(pd.Series([len(e['evidence']) for e in dev_preprocessed]).value_counts())


    # DEV dataset
    qa_noisy_filter_1 = AveritecPre.filter_noisy_same_url(datasets['dev'], qa_noisy)

    for el in datasets['dev']:
        el['ntokens'] = tokenizer_h.ntokens(
            tokenizer_h.strucutre(el['claim'], el['evidence']))  # of the initial <s>

    gold_df = AveritecPre.prepare_qa(datasets['dev'])
    flat_noise_filtered = AveritecPre.prepare_qa(qa_noisy_filter_1)
    # add counter field to mark if an evidence has been added
    flat_noise = AveritecPre.prepare_qa(qa_noisy)
    flat_noise['counter'] = 0

    for el in tqdm(datasets['dev']):
        ev_to_add, b = AveritecPre.get_noisy_evidence(el, gold_df, flat_noise_filtered, flat_noise,
                                                      tokenizer_h.max_length,
                                                      random_seed=random_seed)
        gt_ml = el['ntokens'] > 512
        assert b >= 0 or el['ntokens'] > 512, f'Negative budget {b} for claim {el["claim"]}'
        noise_ev = AveritecPre.convert_ev_df2averitec(ev_to_add)
        el['questions'] = el['questions'] + noise_ev
        el['goldtag'] += [0] * len(noise_ev)
        # shuffle randomly questions and goldtag according to the new evidence
        shuffled_idx = rnd.permutation(len(el['questions']))
        el['questions'] = [el['questions'][i] for i in shuffled_idx]
        el['goldtag'] = [el['goldtag'][i] for i in shuffled_idx]

        #debug
        # el1 = AveritecPre.preprocess_AVERITEC([el], prepend_question=True)[0]
        # el1['ntokens'] = tokenizer_h.ntokens(tokenizer_h.strucutre(el1['claim'], el1['evidence']))
        # assert (el1['ntokens'] > 512) == gt_ml, f'Claim {el1["claim"]} has {el1["ntokens"]} tokens'

    _ = AveritecPre.preprocess_AVERITEC(datasets['dev'], prepend_question=True)
    for el in datasets['dev']:
        el['ntokens'] = tokenizer_h.ntokens(
            tokenizer_h.strucutre(el['claim'], el['evidence']))
    print(pd.Series([len(e['evidence']) for e in datasets['dev']]).value_counts())
    # for el in datasets['dev']:
    #     el['ntokens'] = tokenizer_h.ntokens(tokenizer_h.strucutre(el['claim'], el['evidence']))
    print(pd.Series([el['ntokens'] < 512 for el in datasets['dev']]).value_counts())
    converted_path = os.path.join(xfc.experiment_definitions.E.DATASET_DIR_V3, 'AVERITEC', 'converted')
    os.makedirs(converted_path, exist_ok=True)

    with open(os.path.join(converted_path, 'dev.json'), 'w') as f_out:
        json.dump(datasets['dev'], f_out)

    dev_preprocessed = load_std_dataset(converted_path, 'dev.json')
    print(len(dev_preprocessed))
    print(pd.Series([len(e['evidence']) for e in dev_preprocessed]).value_counts())

