import gc
from typing import List
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import (
    BitsAndBytesConfig)

from explainable_fact_checking.xfc_utils import batched


class LLama3_1Adapter():

    def __init__(self,
                 base_model_name, random_seed, label2id=None, batch_size=8):

        # set random seed
        self.set_random_state(random_seed)
        self.batch_size = batch_size
        if label2id is None:
            label2id = {'NOT ENOUGH INFO': 0, 'SUPPORTS': 1, 'REFUTES': 2}
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        id2label = {v: k for k, v in label2id.items()}

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enables 4-bit quantization
            bnb_4bit_use_double_quant=True,  # Use double quantization for potentially higher accuracy (optional)
            bnb_4bit_quant_type="nf4",  # Quantization type (specifics depend on hardware and library)
            bnb_4bit_compute_dtype=torch.bfloat16  # Compute dtype for improved efficiency (optional)
        )

        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=3,  # Number of output labels (2 for binary sentiment classification)
            id2label=id2label,  # {0: "NEGATIVE", 1: "POSITIVE"}
            label2id=label2id,  # {"NEGATIVE": 0, "POSITIVE": 1}
            quantization_config=bnb_config,  # configuration for quantization
            device_map={"": 0}  # Optional dictionary specifying device mapping (single GPU with index 0 here)
        )

        # adapter_config_path = "/home/bussotti/experiment_AE/0824_explainer_newmodel/llama318b_feverousobj5trained_3epochs/"  #adapter_config.json"
        # adapter_weights_path = "/home/bussotti/experiment_AE/0824_explainer_newmodel/llama318b_feverousobj5trained_3epochs/"  #adapter_model.safetensors"
        adapter_config_path = "/home/bussotti/experiment_AE/0824_explainer_newmodel/llama318b_feverousobj5trained_10epochs_3labels/"  # adapter_config.json"
        adapter_weights_path = "/home/bussotti/experiment_AE/0824_explainer_newmodel/llama318b_feverousobj5trained_10epochs_3labels/"  # adapter_model.safetensors"

        # Load the adapter using the PeftModel class
        self.model = PeftModel.from_pretrained(base_model, adapter_weights_path, adapter_config_path)
        self.softmax = torch.nn.Softmax(dim=1)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def set_random_state(self, random_seed):
        self.random_seed = random_seed
        if random_seed is not None:
            torch.manual_seed(random_seed)

    def predict_batch(self, input: List[str]) -> np.ndarray:
        txt_list = [x['input_txt_to_use'].replace('</s>', '<|reserved_special_token_15|>') for x in input]
        outputs = []
        for t_txt_batch in batched(txt_list, self.batch_size):
            inputs = self.tokenizer(t_txt_batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
            with torch.no_grad():
                try:
                    tout = self.model(**inputs, output_attentions=False)
                except Exception as e:
                    print(f"Error in batch: {e}")
                    print('Trying to recover by cleaning the memory')
                    gc.collect()
                    torch.cuda.empty_cache()
                    tout = self.model(**inputs, output_attentions=False)
                outputs.append(tout['logits'].clone())
                del tout
            del inputs
        logits = self.softmax(torch.cat(outputs, dim=0).cpu().float())

        # batched pred
        # for i, t_txt_batch in enumerate(batched(txt_list, self.batch_size)):
        #     inputs = self.tokenizer(t_txt_batch, return_tensors="pt").to("cuda")
        #     with torch.no_grad():
        #         tout = self.model(**inputs, output_attentions=False, return_dict=True, padding_side="left", padding_max_length=512)
        #
        #    tokenizer.pad_token_id = tokenizer.eos_token_id  # Set a padding token
        #    inputs = tokenizer(texts, padding="longest", return_tensors="pt")
        #    inputs = {key: val.to(model.device) for key, val in inputs.items()}
        #
        #    model.generate(**inputs, max_new_tokens=512, max_length=512, pad_token_id=tokenizer.eos_token_id)

        return logits.numpy(force=True)

    def predict(self, input: List[str]) -> np.ndarray:
        txt_list = [x['input_txt_to_use'].replace('</s>', '<|reserved_special_token_15|>') for x in input]
        outputs = []
        for i, t_txt in enumerate(txt_list):
            inputs = self.tokenizer(t_txt, return_tensors="pt").to("cuda")  # batch predictions are not working

            with torch.no_grad():
                try:
                    tout = self.model(**inputs, output_attentions=False)
                except Exception as e:
                    print(f"Error in iteration {i}.")
                    print(e)
                    print('Trying to recover by cleaning the memory')
                    gc.collect()
                    torch.cuda.empty_cache()
                    tout = self.model(**inputs, output_attentions=False)
                    # prediction with batch
                    # tout = self.model(**inputs, output_attentions=False, return_dict=True
                # pass tout in cpu, create a copy of the logits and append in outputs
                outputs.append(tout['logits'].clone())
                # delete the outputs from the gpu
                del tout
            del inputs
            # if i % 500 == 0:
            #     torch.cuda.empty_cache()
        # logits = torch.cat([torch.sigmoid(x['logits']) for x in outputs], dim=0) # TODO Is softmax the best choice?
        # gc.collect()
        # torch.cuda.empty_cache()
        logits = self.softmax(torch.cat(outputs,
                                        dim=0).cpu().float())  # without float it gives an error `"softmax_lastdim_kernel_impl" not implemented for 'Half'`
        return logits.numpy(force=True)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
