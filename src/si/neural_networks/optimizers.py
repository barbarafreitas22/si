from abc import abstractmethod

import numpy as np


class Optimizer:

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initialize the optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        momentum:
            The momentum to use for updating the weights.
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        return w - self.learning_rate * self.retained_gradient
    

class Adam(Optimizer):
    """"
    Adam (Adaptive Moment Estimation) optimizer.

    This optimizer computes individual adaptive learning rates for different parameters 
    from estimates of first and second moments of the gradients. 
       """

    def __init__(self, learning_rate: float, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize the Adam optimizer with the learning rate and decay parameters.

        Parameters
        ----------
        learning_rate : float
            Step size for updating model weights.
        beta_1 : float, optional
            Exponential decay rate for the first moment estimate (default is 0.9).
        beta_2 : float, optional
            Exponential decay rate for the second moment estimate (default is 0.999).
        epsilon : float, optional
            Small value to prevent division by zero (default is 1e-8).
        """
        super().__init__(learning_rate)
        
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Uses the Adam optimization algorithm to update the weights.

        Parameters
        ----------
        w : np.ndarray
            Model weights.
        grad_loss_w : np.ndarray
            Gradient of the loss function with respect to the weights.

        Returns
        -------
        np.ndarray
            Updated weights after applying Adam optimization.
        """
        # Initialize m and v 
        if self.m is None:
            self.m = np.zeros_like(w)
        if self.v is None:
            self.v = np.zeros_like(w)

        # Increment time step
        self.t += 1

        # Update biased first moment estimate
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_loss_w

        # Update biased second moment estimate
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grad_loss_w ** 2)

        # Compute bias-corrected first moment
        m_hat = self.m / (1 - self.beta_1 ** self.t)

        # Compute bias-corrected second moment
        v_hat = self.v / (1 - self.beta_2 ** self.t)

        # Update weights
        w_updated = w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return w_updated