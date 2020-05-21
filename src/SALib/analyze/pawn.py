import numpy as np
import pandas as pd
from scipy.stats import norm

def analyze(problem, X, Y,
            num_resamples=100,
            conf_level=0.95,
            tuning=10):
    """
    Perform PAWN Analysis on model outputs.

    Returns a dictionary with keys 'S', 'S_conf' where
    each entry is a list of size D (the number of parameters) containing the
    indices in the same order as the parameter file.

    Parameters
    ----------
    problem : dict
        The problem definition
    Y : numpy.array
        A NumPy array containing the model outputs
    num_resamples : int
        The number of resamples (default 100)
    conf_level : float
        The confidence interval level (default 0.95)
    tuning : int
        The number of intervals (default 10)
    """
    if conf_level < 0 or conf_level > 1:
        raise RuntimeError("Confidence level must be between 0-1.")

    if not isinstance(tuning, int):
        raise TypeError("Tuning parameter must be a positive integer.")

    
    
    pass


def cdf():
    pass

def sample_split():
    pass

def KS():
    pass