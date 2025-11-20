import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse 


class RidgeRegressionLeastSquares(Model):
    """
    Ridge Regression implemented using the Least Squares (Normal Equation) method 
    with L2 regularization, providing a closed-form solution for model parameters.
    """

    def __init__(self, l2_penalty: float = 1.0, scale: bool = True, **kwargs):
        """Initializes the Ridge Regression model."""
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
        after handling data scaling.
        """
        X = dataset.X
        y = dataset.y

        # Scale the data and store scaling parameters
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1e-10 # prevent division by zero
        
        if self.scale:
            X = (X - self.mean) / self.std
        
        n_samples, n_features = X.shape

        # Add intercept term to X (column of ones)
        X_ext = np.c_[np.ones(n_samples), X] 

        # Calculate Penalty Term (lambda * I*)
        penalty_term = self.l2_penalty * np.eye(n_features + 1)
        
        # Theta_zero not penalize
        penalty_term[0, 0] = 0.0

        # Normal Equation: theta_all = (X^T @ X + lambda * I*)^-1 @ X^T @ y
        
        # A = (X^T @ X + lambda * I*)
        A = X_ext.T.dot(X_ext) + penalty_term

        # B = X^T @ y
        B = X_ext.T.dot(y)

        # Compute theta_all = A^-1 @ B
        theta_all = np.linalg.inv(A).dot(B)

        # Store parameters
        self.theta_zero = theta_all[0] 
        self.theta = theta_all[1:]    
        self.is_fitted = True
        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the dependent variable (y) for a given dataset.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        X = dataset.X.copy() 

        # Scale the data using stored mean/std (if required)
        if self.scale:
            X = (X - self.mean) / self.std

        n_samples = X.shape[0]

        # Add intercept term to X
        X_ext = np.c_[np.ones(n_samples), X]

        # Concatenate theta_zero and theta
        theta_all = np.r_[self.theta_zero, self.theta]

        # Compute predictions
        predicted_y = X_ext.dot(theta_all)

        return predicted_y

    def score(self, dataset: Dataset) -> float:
        """
        Calculates the Mean Squared Error (MSE) between the real and predicted y values.
        """

        y_pred = self.predict(dataset)
        return mse(dataset.y, y_pred)