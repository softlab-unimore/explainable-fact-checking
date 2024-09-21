import gc

import numpy as np
import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification
from datasets import Dataset, DatasetDict


class RobertaWrapper():

    def __init__(self, model_path, nb_label, random_seed=42, batch_size=256):
        """

        Parameters
        ----------
        model_path : str
            The path of the model
        nb_label : int
            The number of label in dataset
        """
        self.tokenizer = AutoTokenizer.from_pretrained('ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli')
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_path,  # "google/gemma-2b-it"
            num_labels=nb_label,  # Number of output labels (2 for binary sentiment classification)
            device_map={"": 0},  # Optional dictionary specifying device mapping (single GPU with index 0 here)
        )
        self.random_seed = random_seed
        self.batch_size = batch_size

    def predict(self, input):
        # dataset_dict = dataset_dict.map(preprocess_function, batched=True)
        # input_pre = self.tokenizer(input["text"], truncation=False, padding='max_length', max_length=512)
        # dataset_dict = input
        # def filter_long_sequences(example):
        #     return len(example['input_ids']) <= 512

        # Apply the filter to each dataset in the DatasetDict
        # dataset_dict_filtered = DatasetDict({
        #     'train': dataset_dict['train'].filter(filter_long_sequences),
        # })

        # ids_filtered = list(dataset_dict_filtered['train']['ids'])
        # # print(len(ids_filtered))
        # # print(len(ids_original))
        # dataset_dict_full = dataset_dict
        # dataset_dict = dataset_dict_filtered
        #
        # dataset_dict.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'evs_labels'])
        # inputs = self.tokenizer(dataset_dict['train']['text'], return_tensors="pt").to(
        #     'cuda')  # Convert to PyTorch tensors and move to GPU (if available)
        # with torch.no_grad():
        #     outputs = self.model(**inputs)
        #
        # inputs = self.tokenizer(input, return_tensors="pt").to(
        #     "cuda")  # Convert to PyTorch tensors and move to GPU (if available)
        # with torch.no_grad():
        #     outputs = self.model(**inputs).logits  # Get the model's output logits
        # y_prob = torch.sigmoid(outputs).tolist()[0]  # Apply sigmoid activation and convert to list
        txt_list = []
        for elt in input:
            claim_and_ev = ' </s> '.join([elt['claim']] + elt['evidence'])
            txt_list += [claim_and_ev]
        out_list = []
        for batch in range(0, len(txt_list), self.batch_size):
            inputs = self.tokenizer(txt_list[batch:batch+self.batch_size], return_tensors="pt", truncation=False, padding='longest', max_length=512).to(
                'cuda') # Convert to PyTorch tensors and move to GPU (if available)
            with torch.no_grad():
                try:
                    outputs = self.model(**inputs)
                except Exception as e:
                    print(f"Error in batch: {e}")
                    print('Trying to recover by cleaning the memory')
                    gc.collect()
                    torch.cuda.empty_cache()
                    outputs = self.model(**inputs)
                out_list += torch.sigmoid(outputs.logits.clone()).tolist()
            del outputs
            del inputs

        # with torch.no_grad():
        #     outputs = self.model(**inputs)
        # y_prob = torch.sigmoid(outputs.logits).tolist()
        y_prob = out_list
        return np.round(y_prob, 5) # Round the predicted probability to 5 decimal places

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


def main(input_file, input_model, output_file, nb_label):
    tokenizer = AutoTokenizer.from_pretrained('ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli')

    # print(f' Vocab size of the model : {len(tokenizer.get_vocab())}')

    f = open(input_file)
    obj_tr = json.load(f)
    f.close()

    train_list = {'text': [], 'label': [], 'evs_labels': [], 'ids': []}
    for elt in obj_tr:
        claim_and_ev = elt['claim']
        for ev in elt['evidence']:
            claim_and_ev += '</s>' + ev

        train_list['text'] += [claim_and_ev]

        train_list['label'] += [elt['label']]
        train_list['evs_labels'] += [elt['goldtag']]
        train_list['ids'] += [elt['id']]

    train_dataset = Dataset.from_dict(train_list)

    dataset_dict = DatasetDict({
        'train': train_dataset,
    })

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=False, padding='max_length', max_length=512)

    dataset_dict = dataset_dict.map(preprocess_function, batched=True)

    def filter_long_sequences(example):
        return len(example['input_ids']) <= 512

    ids_original = list(dataset_dict['train']['ids'])
    # Apply the filter to each dataset in the DatasetDict
    dataset_dict_filtered = DatasetDict({
        'train': dataset_dict['train'].filter(filter_long_sequences),
    })

    ids_filtered = list(dataset_dict_filtered['train']['ids'])
    # print(len(ids_filtered))
    # print(len(ids_original))
    dataset_dict_full = dataset_dict
    dataset_dict = dataset_dict_filtered

    dataset_dict.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'evs_labels'])

    model = RobertaForSequenceClassification.from_pretrained(
        input_model,  # "google/gemma-2b-it"
        num_labels=nb_label,  # Number of output labels (2 for binary sentiment classification)
        device_map={"": 0},  # Optional dictionary specifying device mapping (single GPU with index 0 here)
    )

    def predict(input_text):

        inputs = tokenizer(input_text, return_tensors="pt").to(
            "cuda")  # Convert to PyTorch tensors and move to GPU (if available)
        with torch.no_grad():
            outputs = model(**inputs).logits  # Get the model's output logits
        y_prob = torch.sigmoid(outputs).tolist()[0]  # Apply sigmoid activation and convert to list
        return np.round(y_prob, 5)  # Round the predicted probability to 5 decimal places

    dataset_dict_tu = dataset_dict_filtered  # dataset_dict_full

    #########EVALUATE
    # outputs_eval=[]

    # gold_evs_labels=[]
    # labels=[]
    predictions = dict()
    # pred_evs_labels=[]
    for i in tqdm(range(len(dataset_dict_tu['train']['text']))):
        # output_tmp=dict()
        # output_tmp['text']=dataset_dict_tu['train']['text'][i]
        inputs = tokenizer(dataset_dict_tu['train']['text'][i], return_tensors="pt").to(
            'cuda')  # Convert to PyTorch tensors and move to GPU (if available)
        with torch.no_grad():
            outputs = model(**inputs)

        predictions[ids_filtered[i]] = outputs.logits.cpu().tolist()
        # labels+=[dataset_dict_tu['train']['label'][i].tolist()]

        # output_tmp['label']=dataset_dict_tu['train']['label'][i].tolist()
        # output_tmp['pred_label']=np.argmax(outputs.logits.cpu(), axis=1)
        # outputs_eval+=[output_tmp]
    predictions_fin = []
    for id_o in ids_original:
        if id_o in ids_filtered:
            predictions_fin += predictions[id_o]
        else:
            # print('there')
            predictions_fin += [[]]
    f = open(output_file, 'w')
    json.dump([predictions_fin], f)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input and output file names.")
    parser.add_argument("--input", help="The name of the input file.")
    parser.add_argument("--input_model", help="The path of the input model .")

    parser.add_argument("--output", help="The name of the output file.")
    parser.add_argument("--nb_label", help="The number of label in dataset.")

    args = parser.parse_args()

    # Pass the arguments to the main function
    main(args.input, args.input_model, args.output, int(args.nb_label))
