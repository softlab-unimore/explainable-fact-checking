import json
import os

import explainable_fact_checking.xfc_utils


def feverous_loader(dataset_dir, dataset_file, top=None, **kwargs):
    data = []
    early_stop = top is not None
    with open(os.path.join(dataset_dir, dataset_file), 'r') as file:
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



