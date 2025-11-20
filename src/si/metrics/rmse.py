import numpy as np
from typing import Union


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Root Mean Squared Error (RMSE) between true values and the predicted ones.

    The RMSE is calculated as: sqrt(mean((y_true - y_pred)^2))

    Parameters
    ----------
    y_true: np.ndarray
        Real values of y.
    y_pred: np.ndarray
        Predicted values of y.

    Returns
    -------
    float
        The RMSE error between y_true and y_pred.
    """
    # Calculate the difference and square it
    error = y_true - y_pred
    squared_error = error ** 2
    
    # Mean of the squared errors
    mean_squared_error = np.mean(squared_error)
    
    root_mean_squared_error = np.sqrt(mean_squared_error)
    
    return root_mean_squared_error