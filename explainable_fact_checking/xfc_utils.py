import json
import logging
import os
import sys

import numpy as np
import scipy.stats as st

class_names = ['NOT ENOUGH INFO', 'SUPPORTS', 'REFUTES']


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


# class to read jsonl files and predict the labels with the feverous model then takes the value of 'input_txt_to_use' for each prediction and add it to the record and save the new file
class AddInputTxtToUse:
    """
        This class provides a static method to predict and save the results using a given model.

        Methods
        -------
        predict_and_save(input_file: str, output_file: str, model: object)
            Predicts the labels for the records in the input file using the provided model and saves the results in the output file.
        """

    @staticmethod
    def predict_and_save(input_file, output_file, model):
        """
        Predicts the labels for the records in the input file using the provided model and saves the results in the output file.

        Parameters
        ----------
        input_file : str
            The path to the input file. The file should be in JSONL format, with each line being a separate JSON object representing a record.
        output_file : str
            The path to the output file where the results will be saved. The results are saved in JSONL format, with each line being a separate JSON object representing a record with the added 'input_txt_to_use' and 'claim' fields.
        model : object
            The model used for prediction. The model should have a 'predict' method that takes a list of records and returns a list of predictions. It should also have a 'predictions' attribute that stores the full predictions data.

        Returns
        -------
        None
        """
        with open(input_file, 'r') as f_in:
            record_list = [json.loads(line) for line in f_in]
            predictions = model.predict_legacy(record_list)
            full_predictions = model.predictions
            for i, pred in enumerate(full_predictions):
                record_list[i]['input_txt_to_use'] = pred['input_txt_model']
                record_list[i]['claim'] = pred['claim']
        with open(output_file, 'w') as f_out:
            for record in record_list:
                f_out.write(json.dumps(record) + '\n')


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
            if 'input_txt_to_use' in record:
                del record['input_txt_to_use']
        predictions = model.predict(record_list)
        full_predictions = model.predictions
        for record, pred in zip(record_list, full_predictions):
            record['input_txt_to_use'] = pred['input_txt_model']
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
        if 'input_txt_to_use' in record:
            ev_list = record['input_txt_to_use'].split(' </s> ')
            n_ev = len(ev_list) -1
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

    def handle_exception(exc_type, exc_value, exc_traceback):
        # Handle exception
        if issubclass(exc_type, KeyboardInterrupt):
            # Call the default KeyboardInterrupt handler
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        # Then propagate the exception
        raise exc_value

    sys.excepthook = handle_exception

    return logger



class GeneralFactory:
    def __init__(self):
        self._creators = {}

    def register_creator(self, name, creator):
        self._creators[name] = creator

    def create(self, name, **kwargs):
        if name not in self._creators:
            raise ValueError(f'The name specified ({name}) is not registered. Valid options are {self._creators.keys()}')
        creator = self._creators[name]
        return creator(**kwargs)