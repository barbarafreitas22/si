import numpy as np


def tanimoto_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculates the Tanimoto distance between a single binary sample (x) 
    and a set of multiple binary samples (y).

    It returns the Tanimoto DISTANCE (1 - Similarity) for each sample in y.

    Parameters
    ----------
    x: np.ndarray
        Single binary sample (1D array).
    y: np.ndarray
        Set of multiple binary samples (2D array).

    Returns
    -------
    np.ndarray
        Tanimoto distance for each sample in y.
    """
    dot_product = np.dot(y, x)

    # binary vectors: sum of elements is equivalent to squared norm
    norm_x_sq = np.sum(x)
    norm_y_sq = np.sum(y, axis=1)

    denominator = norm_x_sq + norm_y_sq - dot_product
    
    # case when denominator is 0
    denominator[denominator == 0] = 1.0 

    similarity = dot_product / denominator
    
    return 1 - similarity









