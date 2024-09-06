import ast
import json
import os
from urllib.request import urlretrieve

import pandas as pd

import explainable_fact_checking as xfc
import explainable_fact_checking.xfc_utils
from explainable_fact_checking import C

# Create an instance of the factory
dataset_loader_factory = explainable_fact_checking.xfc_utils.GeneralFactory()



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
                AddInputTxtToUse.by_model_legacy_prediction(input_file=input_file,
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


class GeneralDatasetLoader:
    file_names = None
    base_link = None

    def download(self, dest_dir):
        '''
        Download the dataset files from the base link to the destination directory.
        '''
        os.makedirs(dest_dir, exist_ok=True)
        for file_name in self.file_names:
            url = self.base_link + file_name + '?raw=true'
            urlretrieve(url, os.path.join(dest_dir, file_name))

    def load(self, dataset_dir, dataset_file, top=None, **kwargs):
        input_file = os.path.join(dataset_dir, dataset_file)
        if dataset_file not in self.file_names:
            raise ValueError(f"Dataset file should be one of {self.file_names}. Got {dataset_file}")
        if not os.path.exists(input_file):
            self.download(dataset_dir)
        # load dataframe from tsv
        df = pd.read_csv(input_file, sep="\t", nrows=top)
        # convert to list of dictionaries
        return self.convert_to_dict(df)

    def convert_to_dict(self, df):
        '''
        Convert the dataframe to a list of dictionaries.
        '''
        return df.to_dict(orient='records')



class PolitihopDatasetLoader(GeneralDatasetLoader):
    '''
    https://github.com/copenlu/politihop/tree/master
    '''
    file_names = ['politihop_train.tsv', 'politihop_valid.tsv', 'politihop_test.tsv']
    base_link = 'https://raw.githubusercontent.com/copenlu/politihop/master/data/'

    def convert_to_dict(self, df):
        '''
        'article_id' (int) -> id
            article id corresponding to the id of the claim in the LIAR dataset
        'statement' (str) -> claim
            the text of the claim
        'author' (str) -> SKIP # todo ok?
            the author of the claim
        'ruling' (List(str)) -> evidence_list [use .apply(ast.literal_eval) to convert to list]
            a comma-separated list of the sentences in the Politifact ruling report (this excludes sentences from the summary in the end)
        'url_sentences' -> SKIP
            a comma-separated list of ids of the sentences with a corresponding source url(s)
        'relevant_text_url_sentences' -> SKIP
            a comma-separated list of ids of the sentences with a corresponding source url(s) that are actually
            relevant to the selected evidence sentences
        'politifact_label' categorical  -> SKIP
            label assigned to the claim by PolitiFact fact-checkiers (https://www.politifact.com/) Truth-o-Meter
            [True, Mostly True, Half True, Mostly False, False, Pants on Fire]
        'annotated_evidence' -> SKIP
            a json dict of the evidence chains (keys) and the sentences that belong to the chain (value, which is a list of sentence ids from the ruling)
        'annotated_label' categorical [True, False, Half-True] -> label
            label annotated by annotators of PolitiFact - True, False, Half-True
        'urls' -> SKIP
            a comma-separated list of source urls used in the corresponding PolitiFact article
        'annotated_urls' -> SKIP
            a json dict mapping sentence ids to the corresponding urls ids. One sentence can have multiple urls
        '''
        # drop columns that are not needed
        df = df.drop(columns=['author', 'url_sentences', 'relevant_text_url_sentences', 'politifact_label',
                              'annotated_evidence', 'urls', 'annotated_urls'])
        # convert ruling to list by applying ast.literal_eval
        df['ruling'] = df['ruling'].apply(ast.literal_eval)
        # rename columns
        df = df.rename(columns=C.COLUMNS_MAP)
        return df.to_dict(orient='records')


dataset_loader_factory.register_creator('politihop', PolitihopDatasetLoader().load)


class LIARPlusDatasetLoader(GeneralDatasetLoader):
    '''
        https://github.com/Tariq60/LIAR-PLUS/tree/master
    '''
    file_names = ['test2.tsv', 'val2.tsv', 'train2.tsv']
    base_link = 'https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/master/dataset/tsv/'

    def load(self, dataset_dir, dataset_file, top=None, **kwargs):
        '''

        Parameters
        ----------
        dataset_dir : str
            The directory where the dataset files are stored.
        dataset_file : str
            The name of the dataset file to load.
        top : int
            The number of records to load from the dataset file. If None, load all records.
        kwargs : dict
            Additional keyword arguments. Not used.


        Returns
        -------
        pd.DataFrame
            The dataset as a pandas dataframe.
            Columns:
                Column 1: the ID of the statement ([ID].json).
                Column 2: the label.
                Column 3: the statement.
                Column 4: the subject(s).
                Column 5: the speaker.
                Column 6: the speaker's job title.
                Column 7: the state info.
                Column 8: the party affiliation.
                Columns 9-13: the total credit history count, including the current statement.
                9: barely true counts.
                10: false counts.
                11: half true counts.
                12: mostly true counts.
                13: pants on fire counts.
                Column 14: the context (venue / location of the speech or statement).
                Column 15: the extracted justification
        '''
        input_file = os.path.join(dataset_dir, dataset_file)
        if dataset_file not in self.file_names:
            raise ValueError(f"Dataset file should be one of {self.file_names}. Got {dataset_file}")
        if not os.path.exists(input_file):
            self.download(dataset_dir)
        columns = ['ID', 'label', 'claim', 'subject', 'speaker', 'job_title', 'state_info', 'party_affiliation',
                   'barely_true', 'false', 'half_true', 'mostly_true', 'pants_on_fire', 'context', 'evidence_list']
        df = pd.read_csv(input_file, sep="\t", nrows=top, header=None, names=columns)

        # convert to list of dictionaries
        return self.convert_to_dict(df)


dataset_loader_factory.register_creator('LIARPlus', LIARPlusDatasetLoader().load)

