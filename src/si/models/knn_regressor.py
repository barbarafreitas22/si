from typing import Callable, Union

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor(Model):
    """
    The K-Nearest Neighbors algorithm adapted for regression tasks.
    It estimates the value based on the average of the K nearest training examples.
    """

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        """
        Initialize the KNN regressor
        """
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        Stores the training dataset (Step 1 of fit)
        """
        self.dataset = dataset
        return self

    def _get_k_nearest_value(self, sample: np.ndarray) -> float:
        """
        Method to get the predicted value for a single sample
        """
        distances = self.distance(sample, self.dataset.X)
        k_nearest_neighbors_indices = np.argsort(distances)[:self.k]
        k_nearest_neighbors_values = self.dataset.y[k_nearest_neighbors_indices]
        prediction = np.mean(k_nearest_neighbors_values)
        
        return prediction

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Estimates the label value for samples based on the k most similar examples.
        """
        # Apply the _get_k_nearest_value method to each sample in the dataset
        predictions = np.apply_along_axis(self._get_k_nearest_value, axis=1, arr=dataset.X)
        
        return predictions

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculates the error between the estimated values and the real ones (using rmse).
        """
        return rmse(dataset.y, predictions)