import numpy as np
import scipy.stats as st


class_names = ['NOT ENOUGH INFO', 'SUPPORTS', 'REFUTES']

# st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))

def mean_confidence_interval(data, confidence=0.95):
    """
    Calculate the confidence interval of the mean of data
    https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data

    :param data: data to calculate the confidence interval
    :param confidence: confidence level, default 0.95
    :return: confidence interval width.
    """
    min, max = st.t.interval(confidence, len(data) - 1, loc=np.mean(data), scale=st.sem(data))
    return (max - min)