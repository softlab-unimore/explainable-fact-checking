rename_word_dict = {
    'only_claim': 'claim alone',
    'claim_intercept': 'CLAIM',
    'claim_only_pred': 'claim isolation',
    'models_fromjf270623or': '2-label',
    'feverous_verdict_predictor': '3-label',
    'score_on_predicted_label': 'contribution on predicted class',
    'predicted_label': 'predicted class',
    'NOT ENOUGH INFO': 'NEI',
    'SUPPORTS': 'SUP',
    'REFUTES': 'REF',
    'feverous_dev_ST_01.jsonl': 'Sentence & Table',
    'feverous_dev_SO_01.jsonl': 'Sentence only',
    'original_TO_01_formatted.jsonl': 'Table only',
    'feverous_train_challenges_withnoise.jsonl': 'With noise',
    'ground_truth': 'GT',
    'explainer_name': 'Explainer',
    'rank_score': 'Rank position',
    'unit_index': 'Unit index',
    'lime': 'LIME',
    'shap': 'SHAP',
    'GenFCExp': 'GFCE',
    'GenFCExp_v2': 'GFCE_2',
    'Roberta': 'RoBERTa',
    'Roberta_v2': 'Rob_2',
    'LLAMA31_70B': 'LLaMa',
    'Roberta_v2_no_noise': 'RoBERTa no noise',
    'AVERITEC': 'AVTC',
    'feverous2l_full': 'Fev.2L',
    'feverous3l_full': 'Fev.3L',
    'FM2': 'FM2',
    'SciFact': 'SciFact',
    'evidence': 'USEFUL',
    'noise': 'NOISE',
    'dataset_name': 'Dataset',
    'model_name': 'Model',

}

key_to_pop = [k for k in rename_word_dict.keys() if k == rename_word_dict[k]]
for k in key_to_pop:
    rename_word_dict.pop(k)

remove_word_list = ['explainer_name=', 'model_id=', 'dataset_file_name=', 'Explainer=', 'model_name=',
                    'dataset_name=', 'Dataset=','Model=' ]


# function to rename words in sentences by first splitting the sentence into words and then joining the words after using a dictionary to replace the words
def replace_words(sentence):
    """
    Replace words in a sentence using a dictionary
    :param sentence: sentence to replace words
    :return: sentence with words replaced
    """
    if sentence is None:
        return None
    for word in remove_word_list:
        sentence = sentence.replace(word, '')
    changed = True
    while changed:
        changed = False
        words = sentence.split(' ')
        for i in range(len(words)):
            if words[i] in rename_word_dict.keys():
                words[i] = rename_word_dict[words[i]]
                changed = True
        sentence = ' '.join(words).strip()
    return sentence
