import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares(Model):
    """
    Ridge Regression with Least Squares (Normal Equation) solves the linear regression problem 
    with L2 regularization using the analytical closed-form solution. It minimizes the sum of 
    squared residuals combined with a penalty on the size of the coefficients (L2 norm),
    which helps reducing overfitting.
    """

    def __init__(self, l2_penalty: float = 1.0, scale: bool = True, **kwargs):
        super().__init__(**kwargs)
        if l2_penalty < 0:
            raise ValueError("l2_penalty must be non-negative.")
            
        self.l2_penalty = l2_penalty
        self.scale = scale
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def _fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Estimates the model parameters (theta and theta_zero) using the Normal Equation 
        after data scaling.
        """
        X = dataset.X
        y = dataset.y

        # Scale the data and store scaling parameters
        if self.scale:
            self.mean = np.nanmean(X, axis=0)
            self.std = np.nanstd(X, axis=0)
            # Prevent division by zero logic
            self.std[self.std == 0] = 1e-10 
            X = (X - self.mean) / self.std
        
        n_samples, n_features = X.shape

        # Add intercept term to X
        # Add a column of 1s to the beginning of matrix X. This allows to calculate 
        # theta_zero (intercept) using matrix algebra. X_ext shape becomes: (n_samples, n_features + 1)
        X_ext = np.c_[np.ones(n_samples), X] 

        # Calculate Penalty Term (lambda * I)
        # This is the "Ridge" part (L2 Regularization
        # # We create a diagonal matrix with the penalty value (lambda).
        penalty_term = self.l2_penalty * np.eye(n_features + 1)
        
        # Theta_zero is not penalized, so the first diagonal element is 0
        penalty_term[0, 0] = 0.0

        # Normal Equation: theta_all = (X^T @ X + lambda * I)^-1 @ X^T @ y
        
        # The Matrix to be inverted (Correlation + Regularization)
        # (X^T @ X) describes the data, (+ penalty_matrix) adds the Ridge constraint.
        A = X_ext.T.dot(X_ext) + penalty_term

        # The Projection of y onto X
        B = X_ext.T.dot(y)

        # Compute the final weights (thetas) by solving the system
        theta_all = np.linalg.inv(A).dot(B)

        # Store parameters
        self.theta_zero = theta_all[0] 
        self.theta = theta_all[1:]
        
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the dependent variable (y) for a given dataset.
        """
        # Note: The 'Model' class already checks if the model is fitted before calling this method

        X = dataset.X
        
        # Scale the data using stored mean/std (if required)
        if self.scale:
            X = (X - self.mean) / self.std

        n_samples = X.shape[0]

        # Add intercept term to X
        X_ext = np.c_[np.ones(n_samples), X]

        # Concatenate theta_zero and theta to rebuild theta_all
        theta_all = np.r_[self.theta_zero, self.theta]

        # Compute predictions
        predicted_y = X_ext.dot(theta_all)

        return predicted_y

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculates the Mean Squared Error (MSE) between the real and predicted y values.
        """
        return mse(dataset.y, predictions)