from  explainable_fact_checking import C

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
            e['evidence_txt_list'] = []
            if 'questions' in e:
                for q in e['questions']:
                    t_txt = ''
                    if prepend_question:
                        t_txt = q['question'] + '?' if not q['question'].endswith('?') else q['question']
                        t_txt += ' '
                    for a in q['answers']:
                        t_txt += a['answer']
                        if a['answer_type'] == 'Boolean' and 'boolean_explanation' in a:
                            t_txt += '. ' + a['boolean_explanation']
                    t_txt = t_txt.replace('\n', ' ')
                    t_txt = t_txt.replace('  ', ' ')
                    e[C.EV_KEY].append(t_txt)
    return dataset