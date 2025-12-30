from abc import abstractmethod

import numpy as np


class LossFunction:

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the loss function for a given prediction.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the derivative of the loss function for a given prediction.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """
    Mean squared error loss function.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the mean squared error loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the mean squared error loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        # To avoid the additional multiplication by -1 just swap the y_pred and y_true.
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(LossFunction):
    """
    Cross entropy loss function.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true / p) + (1 - y_true) / (1 - p)

class CategoricalCrossEntropy(LossFunction):
    """
    Computes the cross-entropy loss between true labels and predicted labels.

    It quantifies the difference between two probability distributions: the true distribution 
    (100% for the correct class) and the predicted distribution.
    """

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the mean loss value for the given predictions.

        The calculation involves determining the negative log-likelihood of the 
        true class. To prevent numerical instability, the predictions are clipped 
        to a range.

        Parameters
        ----------
        y_true : np.ndarray
            The ground truth labels. 
            Shape: (n_samples, n_classes).
        y_pred : np.ndarray
            The predicted probability distribution for each sample.
            Shape: (n_samples, n_classes).

        Returns
        -------
        float
            The scalar loss value, averaged over all samples in the batch.
        """
        # Ensure numerical stability by clipping predictions
        clipped_predictions = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Compute the cross-entropy loss
        sum_loss = -np.sum(y_true * np.log(clipped_predictions))
        
        return sum_loss / y_true.shape[0]

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the loss with respect to the predictions.

        This gradient indicates the direction and magnitude that predictions 
        need to change to minimize the error. 
        Parameters
        ----------
        y_true : np.ndarray
            The truth labels.
        y_pred : np.ndarray
            The predicted probabilities.

        Returns
        -------
        np.ndarray
            A matrix of the same shape as y_pred containing the gradients.
        """
        # Ensure numerical stability 
        clipped_predictions = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Compute the gradient
        return - (y_true / clipped_predictions) / y_true.shape[0]