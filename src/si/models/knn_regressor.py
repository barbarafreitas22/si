from typing import Callable, Union
import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor(Model):
    """
    The K-Nearest Neighbors algorithm adapted for regression tasks, estimating 
    the value based on the average of the K nearest training examples.
    """

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        """
        Initialize the KNN regressor with the number of neighbors (k) and distance function.

        Parameters
        ----------
        k : int, default=1
            Number of nearest neighbors to consider for prediction.
        distance : Callable, default=euclidean_distance
            Function to compute the distance between samples.
        """
        super().__init__(**kwargs)
        if k < 1:
            raise ValueError("k must be greater than or equal to 1.")
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset: Dataset) -> "KNNRegressor":
        """
        Store the training dataset for future predictions.

        Parameters
        ----------
        dataset : Dataset
            The training data.

        Returns
        -------
        KNNRegressor
            The fitted model instance.
        """
        self.dataset = dataset
        self.is_fitted = True
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict target values for a test dataset by averaging the values of the 
        k-nearest neighbors for each sample.

        Parameters
        ----------
        dataset : Dataset
            The test dataset to predict target values for.

        Returns
        -------
        np.ndarray
            Array of predicted target values (y_pred).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")

        X_train = self.dataset.X
        y_train = self.dataset.y
        X_test = dataset.X
        
        predictions = []

        for sample in X_test:
            # Distances from the sample to all training samples
            distances = self.distance(sample, X_train)
            
            # Identify the indices of the k closest neighbors
            nearest_indices = np.argsort(distances)[:self.k]
            
            # Retrieve their target values and calculate the prediction
            neighbor_values = y_train[nearest_indices]
            prediction = np.mean(neighbor_values)
            
            predictions.append(prediction)
        return np.array(predictions)

    def _score(self, dataset: Dataset) -> float:
        """
        Evaluate the model using Root Mean Squared Error (RMSE).

        Parameters
        ----------
        dataset : Dataset
            The dataset with true target values (y_true).

        Returns
        -------
        float
            RMSE value indicating the prediction error.
        """
        y_pred = self.predict(dataset)
        
        # Retrieve the prediction true values and calculate RMSE
        y_true = dataset.y
        return rmse(y_true, y_pred)