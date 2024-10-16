import numpy as np

import explainable_fact_checking as xfc
from explainable_fact_checking.model_adapters.genFCExp import GenFCExp
from explainable_fact_checking.model_adapters.roberta import RobertaWrapper


# models_dict = {
#     'default': FeverousModelAdapter,
# }


# def get_model(model_name, random_state=42, **kwargs):
#     model_class = models_dict.get(model_name, None)
#     if model_class is None:
#         raise ValueError(
#             f'the method specified ({model_name}) is not allowed. Valid options are {models_dict.keys()}')
#
#     class PersonalizedWrapper:
#         def __init__(self, method_str, random_state=42, **kwargs):
#             self.method_str = method_str
#             self.model = model_class(random_state=random_state, **kwargs)
#
#     # params = inspect.signature(model_class.predict).parameters.keys()
#     # if 'sensitive_features' in params:
#     #     def predict(self, X, sensitive_features):
#     #         return self.model.predict(X, sensitive_features=sensitive_features)
#     # else:
#     #     def predict(self, X):
#     #         return self.model.predict(X)
#     # PersonalizedWrapper.predict = predict
#
#     return PersonalizedWrapper(model_name, random_state, **kwargs)


class FakePredictor:
    def __init__(self, random_state=42, **kwargs):
        self.random_state = random_state

    # fake predictor function that takes in input a set of list and returns a list of
    # random predictions with shape (len(strings), 1)
    @staticmethod
    def predict(restructured_records):
        # increasing prediction with the length of the evidence
        # predictions = np.array([len(x['evidence'][0]['content']) for x in restructured_records]).reshape(-1, 1)
        # predictions = predictions / np.max(predictions)

        predictions: np.ndarray = np.random.rand(len(restructured_records), 1)
        # scale the predictions between 0 and 1
        predictions = np.concatenate([predictions, 1 - predictions, np.zeros_like(predictions)], axis=1)
        return predictions

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

# Create an instance of the factory
model_factory = xfc.xfc_utils.GeneralFactory()

from explainable_fact_checking.model_adapters.feverous_model import FeverousModelAdapter
model_factory.register_creator('default', FeverousModelAdapter)

from explainable_fact_checking.model_adapters.llm_adapter import LLama3_1Adapter
model_factory.register_creator('LLAMA3_1', LLama3_1Adapter)

model_factory.register_creator('fake_predictor', FakePredictor)

model_factory.register_creator('Roberta', RobertaWrapper)
model_factory.register_creator('Roberta_v2', RobertaWrapper)
model_factory.register_creator('Roberta_v2_no_noise', RobertaWrapper)
model_factory.register_creator('Roberta_bad', RobertaWrapper)

model_factory.register_creator('GenFCExp', GenFCExp)
model_factory.register_creator('GenFCExp_v2', GenFCExp)


from explainable_fact_checking.model_adapters.llama import LLAMA31Wrapper
model_factory.register_creator('LLAMA31_8B', LLAMA31Wrapper)
model_factory.register_creator('LLAMA31_70B', LLAMA31Wrapper)

