import json
import os

import explainable_fact_checking as xfc
import explainable_fact_checking.xfc_utils
from explainable_fact_checking import FeverousModelAdapter, C


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


# dataset_loader_dict = {
#     'feverous': feverous_loader,
# }
#
#
# def get_dataset(dataset_name, **kwargs):
#     loader = dataset_loader_dict.get(dataset_name, None)
#     if loader is None:
#         raise ValueError(
#             f'The dataset specified ({dataset_name}) is not allowed. Valid options are {dataset_loader_dict.keys()}')
#     return loader(**kwargs)
#
# class DatasetLoaderFactory:
#     def __init__(self):
#         self._loaders = {}
#
#     def register_loader(self, dataset_name, loader_function):
#         self._loaders[dataset_name] = loader_function
#
#     def get_dataset(self, dataset_name, **kwargs):
#         loader = self._loaders.get(dataset_name)
#         if loader is None:
#             raise ValueError(
#                 f'The dataset specified ({dataset_name}) is not allowed. Valid options are {self._loaders.keys()}')
#         return loader(**kwargs)


# Create an instance of the factory
dataset_loader_factory = explainable_fact_checking.xfc_utils.GeneralFactory()

# Register the loaders
dataset_loader_factory.register_creator('feverous', feverous_loader)


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
                AddInputTxtToUse.generate_txt_from_list_keys(input_file, output_file=output_file,)
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
