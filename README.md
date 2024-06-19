
# Explainable Fact Checking
Code repository for the paper 2598 at EMNLP 2024.

# Our proposal

<img src="img/Architeture_schema.png" alt="Our Architeture proposal" height="500"/>

The figure shows our framework
to understand the _contribution_ of each evidence in predicting the label for a claim.
It adapts a local post-hoc explainer (LIME and SHAP in our study) to evidence and claim.
Local post-hoc explainers rely on "variations" of the input to compute the contributions of the features 
(every example's evidences) in the decision (for a claim).
In our framework, a perturbed dataset is generated for the example by  removing pieces of evidence. 

The explainers use the predictions of the verifier over the perturbed dataset coupled with the original claim 
to build the explanation. It assigns for each example a contribution to each feature (evidence) 
and to an _intercept_, defined as the average difference between the overall prediction 
and the total contribution of all individual evidences in the perturbed dataset.
This can be interpreted as the contribution that the explainer attributes to the claim 
within the context provided by the evidence. 

The prediction is thus approximated  expressed as a function of the contributions defined by the explanation:

$$Pred \approx \sum_{i=1}^{N} \text{contribution}(e_i) + \text{Intercept}$$

where contribution( $e_i$ ) (contribution of evidence $i$ ) is the contribution over the model
prediction detected by an explainer for an evidence. 


# Data
The data used in this paper is from the FEVER dataset.
This dataset can be downloaded from the [official FEVER website](https://fever.ai/dataset/feverous.html). 
The datasets filtered by evidence type used for the experiments are stored in the `datasets` directory.

# Anonymization
Googling any part of the paper or the online appendix can be considered as a deliberate attempt to break anonymity ([explanation](https://www.monperrus.net/martin/open-science-double-blind))

# API

**Python Interface**
You can define the experiment settings in the form of a Python dictionary and use one of the following
Python functions to run experiments:
    
1. You can define and organize your experiment configurations in a python module as a list of dictionaries
each representing in an experiment set of settings. **This is the reccommended interface.**
You can create your own file or use the default at `explainable_fact_checking.experiment_definitions`).

Here is an example of how to define an experiment configuration:

```python
experiment_definitions = [
    dict(experiment_id='sk_f_jf_1.0',
         results_dir='results_dir',         
         random_seed=[1],
         
         # model params
         model_name=['default'],
         model_params=dict(model_path=['model_dir/model_name']),  # [, param1=[v1, v2]]
         
         # explainer params
         explainer_name=['lime'],
         explainer_params=dict(perturbation_mode=['only_evidence'], num_samples=[500]),
         
         # dataset params
         dataset_name='feverous',
         dataset_params=dict(
             dataset_dir=['dataset_dir'],
             dataset_file=['file1.jsonl', 'file2.jsonl'],
             top=[1000]),
         )
    ]
```


Then to run the experiments you use `explainable_fact_checking.experiment_routine.ExperimentRunner.launch_experiment_by_id`
 E.g.:
``` python
from explainable_fact_checking.experiment_routine import ExperimentRunner

if __name__ == '__main__':
    exp_runner = ExperimentRunner() 
    exp_runner.launch_experiment_by_id('exp_id.1')
```

if you want to run the experiments with your personal configuration file, you can specify  
the path to the file in the `launch_experiment_by_id` function in `config_file_path` parameter.
E.g.:
``` python
# as before, but in the main use.
    exp_runner.launch_experiment_by_id('exp_id.1', config_file_path='path/to/your/config_file.py')
```



2. `fairnesseval.run.launch_experiment_by_config` let you run an experiment by passing the dictionary of parameters
of your experiment in input.
E.g.:
``` python
from explainable_fact_checking.experiment_routine import ExperimentRunner
    
experiment_config = dict(...) # as in the experiment configuration example

if __name__ == '__main__':
    exp_runner = ExperimentRunner() 
    exp_runner.launch_experiment_by_config(experiment_config)
```

## Experiment parameters
This table provides a clear and concise overview of the parameters and their descriptions.

| Parameter | Description                                                                                                                                                                                                                                                                          |
| --- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `experiment_id` | ID of the experiment to run. Required. [str]                                                                                                                                                                                                                                         |
| `results_dir` | Path to save results. [str]                                                                                                                                                                                                                                                          |
| `random_seed` | List of random seeds to use. All random seeds set are related to this random seed. For each random_seed a new train_test split is done.                                                                                                                                              |
| `model_name` | List of model names. Required.                                                                                                                                                                                                                                                       |
| `model_params` | Dict with pairs of model hyper parameter names (key) and list of values to be iterated (values) for the specified models in `model_name`. The cross product of the list of parameters values is used to generate all the combinations of parameters to test.                         |
| `explainer_name` | List of explainer names. Required.                                                                                                                                                                                                                                                   |
| `explainer_params` | Dict with pairs of explainer hyper parameter names (key) and list of values to be iterated (values) for the specified models in `explainer_name`. The cross product of the parameters is used as in `model_params`                                                                   |
| `dataset_name` | List of dataset loader function names. Required.                                                                                                                                                                                                                                     |
| `dataset_params` | Dict with pairs of dataset hyper parameter names (key) and list of values to be iterated (values) for the specified in `dataset_name`. The cross product of the parameters is used as in `model_params`.                                                                                          |


