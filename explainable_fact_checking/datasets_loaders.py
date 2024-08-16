import json
import os
from urllib.request import urlretrieve

import pandas as pd

import explainable_fact_checking as xfc
import explainable_fact_checking.xfc_utils
from explainable_fact_checking import FeverousModelAdapter, C


# Create an instance of the factory
dataset_loader_factory = explainable_fact_checking.xfc_utils.GeneralFactory()


def feverous_loader(dataset_dir, dataset_file, top=None, **kwargs):
    data = []
    input_file = os.path.join(dataset_dir, dataset_file)
    input_file = AddInputTxtToUse.check_dataset(input_file)

    early_stop = top is not None
    with open(input_file, 'r') as file:
        if early_stop:
            for i, line in enumerate(file):
                if i >= top:
                    break
                if line != '\n':
                    data.append(json.loads(line))
        else:
            for i, line in enumerate(file):
                if line != '\n':
                    data.append(json.loads(line))
    return data


# Register the loaders
dataset_loader_factory.register_creator('feverous', feverous_loader)

class PolitihopDatasetLoader:

    file_names = ['politihop_train.tsv', 'politihop_valid.tsv', 'politihop_test.tsv']

    @staticmethod
    def download_politihop(dest_dir):
        '''
        https://github.com/copenlu/politihop/tree/master
        '''
        base_link = 'https://github.com/copenlu/politihop/blob/master/data/'
        os.makedirs(dest_dir, exist_ok=True)
        for file_name in PolitihopDatasetLoader.file_names:
            url = base_link + file_name + '?raw=true'
            # download file using python library urllib
            urlretrieve(url, os.path.join(dest_dir, file_name))

    @staticmethod
    def politihop_loader(dataset_dir, dataset_file, top=None, **kwargs):
        data = []
        input_file = os.path.join(dataset_dir, dataset_file)
        #if datest file not in file_names raise error
        if dataset_file not in PolitihopDatasetLoader.file_names:
            raise ValueError(f"Dataset file should be one of {PolitihopDatasetLoader.file_names}.")
        # if file does not exist, download it
        if not os.path.exists(input_file):
            PolitihopDatasetLoader.download_politihop(dataset_dir)

        # load dataframe from tsv
        df = pd.read_csv(input_file, sep='\t')
        return df


dataset_loader_factory.register_creator('politihop', PolitihopDatasetLoader.politihop_loader)


class AddInputTxtToUse:
    """
        This class provides a static method to predict and save the results using a given model.
        class to read jsonl files and predict the labels with the feverous model then takes the
        value of 'input_txt_to_use' for each prediction and add it to the record and save the new file.

        Methods
        -------
        predict_and_save(input_file: str, output_file: str, model: object)
            Predicts the labels for the records in the input file using the provided model and saves the results in the output file.
        """

    @staticmethod
    def check_dataset(input_file):
        # if file with '_plus.jsonl' exists, use it
        if os.path.exists(new_name := input_file.replace('.jsonl', '_plus.jsonl')):
            return new_name
        with open(input_file, 'r') as file:
            line = file.readline()
            while line == '\n':
                line = file.readline()
            row = json.loads(line)
            # if it has the input txt to use field break. If it is not present, add it with
            if C.TXT_TO_USE in row:
                return input_file
            output_file = input_file.replace('.jsonl', '_plus.jsonl')
            if C.KEYS_TEXT in row:
                AddInputTxtToUse.generate_txt_from_list_keys(input_file, output_file=output_file, )
            else:
                xfc.xfc_utils.AddInputTxtToUse.by_model_legacy_prediction(input_file=input_file,
                                                                          output_file=output_file,
                                                                          )
            return output_file

    @staticmethod
    def generate_txt_from_list_keys(input_file, output_file):
        """
        Generate the 'input_txt_to_use' field from the 'list_keys_and_text_in_order' field in the input file.
        save the new file with the same name.

        Parameters
        ----------
        input_file : str
            The path to the input file. The file should be in JSONL format, with each line being a separate JSON object representing a record.

        Returns
        -------
        None
        """
        assert input_file != output_file, f'Input and Output files are equal when generating {C.TXT_TO_USE}'
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            try:
                record_list = [json.loads(line) for line in f_in]
                for record in record_list:
                    record[C.TXT_TO_USE] = ' </s> '.join(
                        [record['claim']] + [x[1] for x in record[C.KEYS_TEXT]])
                    f_out.write(json.dumps(record) + '\n')
            except Exception as e:
                # delete the output file if an exception is raised
                if os.path.exists(output_file):
                    os.remove(output_file)
                raise e

    @staticmethod
    def by_model_legacy_prediction(input_file, output_file, ):
        """
        Predicts the labels for the records in the input file using the provided model and saves the results in the output file.

        Parameters
        ----------
        input_file : str
            The path to the input file. The file should be in JSONL format, with each line being a separate JSON object representing a record.
        output_file : str
            The path to the output file where the results will be saved. The results are saved in JSONL format, with each line being a separate JSON object representing a record with the added 'input_txt_to_use' and 'claim' fields.
        Returns
        -------
        None
        """
        # default model
        model = xfc.FeverousModelAdapter(
            model_path=xfc.experiment_definitions.C.baseline_feverous_model['model_params']['model_path'][
                0])
        with open(input_file, 'r') as f_in:
            record_list = [json.loads(line) for line in f_in]
            predictions = model.predict_legacy(record_list)
            full_predictions = model.predictions
            for i, pred in enumerate(full_predictions):
                record_list[i][C.TXT_TO_USE] = pred[C.TXT_TO_USE]
                record_list[i]['claim'] = pred['claim']
        with open(output_file, 'w') as f_out:
            for record in record_list:
                f_out.write(json.dumps(record) + '\n')

