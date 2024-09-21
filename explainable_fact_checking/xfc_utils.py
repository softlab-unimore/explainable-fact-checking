import json
import logging
import os
import sys
from itertools import islice

import numpy as np
import scipy.stats as st


class C:  # general costants
    KEYS_TEXT = 'list_keys_and_text_in_order'
    TXT_TO_USE = 'input_txt_to_use'
    EV_KEY = 'evidence_txt_list'
    COLUMNS_MAP = {'article_id': 'id',
                   'statement': 'claim', 'ruling': EV_KEY,
                   'annotated_label': 'label'}


# Root directory
base_path = '/homes/bussotti/feverous_work/feverousdata'

class_names_load = ['NOT ENOUGH INFO', 'SUPPORTS', 'REFUTES']


# st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))

def mean_confidence_interval(data, confidence=0.95):
    """
    Calculate the confidence interval of the mean of data
    https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data

    Parameters
    ----------
    data : list
        The data to calculate the confidence interval.
    confidence : float
        The confidence level of the interval.

    Returns
    -------
    float
        The confidence interval of the mean of data. (max - min)
    """
    min, max = st.t.interval(confidence, len(data) - 1, loc=np.mean(data), scale=st.sem(data))
    return (max - min)


def save_prediciton_only_claim(input_file, output_file, model):
    """
    Predicts the labels for the records without the evidence field from the input file using the provided model and saves the results in the output file.

    Parameters
    ----------
    input_file : str
        The path to the input file. The file should be in JSONL format, with each line being a separate JSON object representing a record.
    output_file : str
        The path to the output file where the results will be saved. The results are saved in JSONL format, with each line being a separate JSON object representing a record without the 'evidence' field.
    model : object
        The model used for prediction. The model should have a 'predict' method that takes a list of records and returns a list of predictions. It should also have a 'predictions' attribute that stores the full predictions.

    Returns
    -------
    None
    """
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        record_list = [json.loads(line) for line in f_in]
        for record in record_list:
            record['evidence'] = []
            # delete the input_txt_to_use field if present
            if C.TXT_TO_USE in record:
                del record[C.TXT_TO_USE]
        predictions = model.predict(record_list)
        full_predictions = model.predictions
        for record, pred in zip(record_list, full_predictions):
            record[C.TXT_TO_USE] = pred['input_txt_model']
            record['claim'] = pred['claim']
            t = 'predicted_scores'
            record[t] = pred[t]
        json.dump(record_list, f_out)


def map_evidence_types(records):
    """
    Populate a dictionary with each example by recording the id and a list of the evidence types.
    The evidence types can be cell or sentence. It is recognized by the presence of the 'cell' key in the evidence dictionary.

    Parameters
    ----------
    records : list
        A list of records, where each record is a dictionary representing an example.

    Returns
    -------
    dict
        A dictionary with the id and a list of evidence types for each example.
    """
    out_dict = {}
    for record in records:
        id = record['id']
        evidence_types = []
        for evidence_dict in record['evidence']:
            for evidence in evidence_dict['content']:
                if '_sentence_' in evidence:
                    evidence_types.append('sentence')
                # elif '_cell_' in evidence:
                #     evidence_types.append('cell')
                else:
                    pass
                    # evidence_types.append('cell')
                    # raise ValueError('Evidence type not recognized.')

        # assert len(np.nunique(evidence_types)) == len(record['evidence']), f'Number of unique evidence types {np.nunique(evidence_types)} different from number of evidence dictionary {len(record["evidence"])}'
        if C.TXT_TO_USE in record:
            ev_list = record[C.TXT_TO_USE].split(' </s> ')
            n_ev = len(ev_list) - 1
            # assert n_ev == len(evidence_types), f'Number of evidence types {len(evidence_types)} different from number of evidence {n_ev}'
        out_dict[id] = evidence_types
    return out_dict


def init_logger(save_dir: str) -> logging.Logger:
    logger = logging.getLogger(__name__)
    # reset the logger
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(os.path.join(save_dir, 'run.log'), mode='w')
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)
    c_format = logging.Formatter(fmt='%(asctime)s %(levelname)s:%(name)s: %(message)s',
                                 datefmt='%d/%m/%y %H:%M:%S', )
    f_format = logging.Formatter(fmt='%(asctime)s %(levelname)s:%(name)s: %(message)s',
                                 datefmt='%d/%m/%y %H:%M:%S', )
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    def except_hook(exc_type, exc_value, exc_traceback):
        # Handle exception
        if issubclass(exc_type, KeyboardInterrupt):
            # Call the default KeyboardInterrupt handler
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        # Then propagate the exception
        raise exc_value

    sys.excepthook = except_hook

    return logger


def handle_exception(e, logger):
    """
    Handle an exception by logging it and raising it again if the code is being debugged.

    Parameters
    ----------
    e : Exception
        The exception to be handled.
    logger : logging.Logger
        The logger to be used for logging the exception.

    Returns
    -------
    None
    """
    logger.error(e)
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        pass
    elif gettrace():
        raise e


class GeneralFactory:
    def __init__(self):
        self._creators = {}

    def register_creator(self, name, creator):
        self._creators[name] = creator

    def create(self, name, **kwargs):
        if name not in self._creators:
            raise ValueError(
                f'''The name specified ({name}) is not registered. Valid options are {self._creators.keys()}.
                Please register the creator before using it. 
                To do so use `register_creator` from `explainable_fact_checking.datasets_loaders.dataset_loader_factory` after importing the library.
                ''')
        creator = self._creators[name]
        return creator(**kwargs)


def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


def is_debugging():
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None
