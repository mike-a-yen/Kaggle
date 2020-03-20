import warnings

import numpy as np


def find_threshold_at_positive_rate(predictions: np.ndarray, positive_rate: float = 1e-3, precision: int = 4) -> float:
    """Find the threshold that does not exceed the positive rate.

    Parameters
    ----------
    predictions : np.ndarray
        Array of prob_is_positive
    positive_rate : np.ndarray
        The allowable rate of predicting a 1.
    precision : int
        The precision of the threshold.
    """
    N = predictions.shape[0]
    num_allowed = int(np.floor(positive_rate * N))
    if num_allowed == 0:
        warnings.warn(f'Not enough samples {N} for positive rate {positive_rate}. Using {num_allowed / N}.')
        num_allowed = 1
    probs = np.around(predictions, decimals=precision)
    ordered_probs = np.sort(probs)
    thresh = ordered_probs[-num_allowed]
    return float(thresh)
