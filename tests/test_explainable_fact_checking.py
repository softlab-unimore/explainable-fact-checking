import json
import os

import shap
from lime.lime_text import LimeTextExplainer

from explainable_fact_checking import FeverousModelAdapter, WrapperExplaniableFactChecking, CustomTokenizer


def test_predict():
    # init FeverousModelAdapter

    AB_path = '/homes/bussotti/feverous_work/feverousdata/AB/'
    example_file = AB_path + 'sub_stcetb.json'

    with open(example_file, 'r') as file:
        data = [json.loads(line) for line in file if line != '\n']

    fc_model = FeverousModelAdapter()
    predictions = fc_model.predict(data)
    assert len(predictions) == len(data)


def test_shap():
    AB_path = '/homes/bussotti/feverous_work/feverousdata/AB/'
    example_file = AB_path + 'sub_stcetb.json'

    with open(example_file, 'r') as file:
        data = [json.loads(line) for line in file if line != '\n']
    record = data[0]

    fc_model = FeverousModelAdapter()
    predictor = fc_model.predict

    fc_exp_wrapper = WrapperExplaniableFactChecking(record, predictor)

    explainer = shap.Explainer(fc_exp_wrapper, masker=CustomTokenizer)
    shap_values = explainer(fc_exp_wrapper.get_text_to_perturb())
    assert len(shap_values) == len(fc_exp_wrapper.get_text_to_perturb().split(fc_exp_wrapper.separator))


def test_lime():
    AB_path = '/homes/bussotti/feverous_work/feverousdata/AB/'
    example_file = AB_path + 'sub_stcetb.json'

    with open(example_file, 'r') as file:
        data = [json.loads(line) for line in file if line != '\n']
    record = data[0]

    fc_model = FeverousModelAdapter()
    predictor = fc_model.predict

    fc_exp_wrapper = WrapperExplaniableFactChecking(record, predictor)

    explainer = LimeTextExplainer(
        split_expression=fc_exp_wrapper.tokenizer,
        # Order matters, and we cannot use bag of words.
        bow=False,
        class_names=['SUPPORTS'  # , 'REFUTES', 'NOT ENOUGH INFO'
                     ],
    )
    exp = explainer.explain_instance(
        text_instance=fc_exp_wrapper.get_text_to_perturb(),
        classifier_fn=fc_exp_wrapper,
        top_labels=1,
        num_features=fc_exp_wrapper.get_num_evidence(),
        num_samples=50,
    )
    assert len(exp.as_map()[0]) == fc_exp_wrapper.get_num_evidence()

# if __name__ == "__main__":
#     test_shap()
