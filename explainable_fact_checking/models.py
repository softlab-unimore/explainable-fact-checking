import numpy as np

import explainable_fact_checking as xfc

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


# fake predictor function that takes in input a set of list and returns a list of
# random predictions with shape (len(strings), 1)
def fake_predictor(restructured_records):
    # increasing prediction with the length of the evidence
    # predictions = np.array([len(x['evidence'][0]['content']) for x in restructured_records]).reshape(-1, 1)
    # predictions = predictions / np.max(predictions)

    predictions: np.ndarray = np.random.rand(len(restructured_records), 1)
    # scale the predictions between 0 and 1
    predictions = np.concatenate([predictions, 1 - predictions, np.zeros_like(predictions)], axis=1)
    return predictions

# Create an instance of the factory
model_factory = xfc.xfc_utils.GeneralFactory()

# Register the models
model_factory.register_creator('default', xfc.FeverousModelAdapter)