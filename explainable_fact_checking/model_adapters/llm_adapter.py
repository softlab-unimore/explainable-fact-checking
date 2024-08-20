import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import (
    BitsAndBytesConfig)


class LLMAdapter():

    def __init__(self,
                 base_model_name, random_seed, label2id=None):
        if label2id is None:
            label2id = {"REFUTES": 0, "SUPPORTS": 1}
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        id2label = {v: k for k, v in label2id.items()}

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enables 4-bit quantization
            bnb_4bit_use_double_quant=True,  # Use double quantization for potentially higher accuracy (optional)
            bnb_4bit_quant_type="nf4",  # Quantization type (specifics depend on hardware and library)
            bnb_4bit_compute_dtype=torch.bfloat16  # Compute dtype for improved efficiency (optional)
        )

        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,  # "google/gemma-2b-it"
            num_labels=2,  # Number of output labels (2 for binary sentiment classification)
            id2label=id2label,  # {0: "NEGATIVE", 1: "POSITIVE"}
            label2id=label2id,  # {"NEGATIVE": 0, "POSITIVE": 1}
            quantization_config=bnb_config,  # configuration for quantization
            device_map={"": 0}  # Optional dictionary specifying device mapping (single GPU with index 0 here)
        )

        # Load the adapter configuration and weights

        ## Andrea : t=
        adapter_config_path = "/home/bussotti/experiment_AE/0824_explainer_newmodel/llama318b_feverousobj5trained_3epochs/"  #adapter_config.json"
        adapter_weights_path = "/home/bussotti/experiment_AE/0824_explainer_newmodel/llama318b_feverousobj5trained_3epochs/"  #adapter_model.safetensors"

        # Load the adapter using the PeftModel class
        self.model = PeftModel.from_pretrained(base_model, adapter_weights_path, adapter_config_path)
        # set random seed
        self.set_random_state(random_seed)

    def set_random_state(self, random_seed):
        self.random_seed = random_seed
        if random_seed is not None:
            torch.manual_seed(random_seed)


    def predict(self, input):
        # inputs = self.tokenizer(
        #     "coucou <|reserved_special_token_15|> How are you?<|reserved_special_token_15|> How are you?<|reserved_special_token_15|> How are you? ",
        #     return_tensors="pt").to("cuda")  # Convert to PyTorch tensors and move to GPU (if available)
        # # apply the tokenizer to the input text list
        inputs = self.tokenizer(input, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        return outputs

    def __call__(self, *args, **kwargs):
        self.predict(*args, **kwargs)
