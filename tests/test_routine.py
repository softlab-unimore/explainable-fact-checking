import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from explainable_fact_checking.experiment_routine import ExperimentRunner


# main thing to start the experiment
def test_routine():
    test_routine = [
        'test_3.0',
        'test_1.0',
        'test_2.0',
        'test_2.1',
        'test_1.1',
    ]

    experiment_runner = ExperimentRunner()
    for exp_id in test_routine:
        experiment_runner.launch_experiment_by_id(exp_id)
