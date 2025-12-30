from abc import ABCMeta, abstractmethod
import copy
from typing import Optional

import numpy as np

from si.neural_networks.optimizers import Optimizer


class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self, input):
        raise NotImplementedError
    
    @abstractmethod
    def backward_propagation(self, error):
        raise NotImplementedError
    
    @abstractmethod
    def output_shape(self):
        raise NotImplementedError
    
    @abstractmethod
    def parameters(self):
        raise NotImplementedError
    
    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape
    
    def layer_name(self):
        return self.__class__.__name__
    
class DenseLayer(Layer):
    """
    Dense layer of a neural network.
    """

    def __init__(self, n_units: int, input_shape: tuple = None):
        """
        Initialize the dense layer.

        Parameters
        ----------
        n_units: int
            The number of units of the layer, aka the number of neurons, aka the dimensionality of the output space.
        input_shape: tuple
            The shape of the input to the layer.
        """
        super().__init__()
        self.n_units = n_units
        self._input_shape = input_shape

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

    def initialize(self, optimizer: Optimizer) -> 'DenseLayer':
        # initialize weights from a 0 centered uniform distribution [-0.5, 0.5)
        self.weights = np.random.rand(self.input_shape()[0], self.n_units) - 0.5
        # initialize biases to 0
        self.biases = np.zeros((1, self.n_units))
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output
    
    def backward_propagation(self, output_error: np.ndarray) -> float:
        """
        Perform backward propagation on the given output error.
        Computes the dE/dW, dE/dB for a given output_error=dE/dY.
        Returns input_error=dE/dX to feed the previous layer.

        Parameters
        ----------
        output_error: numpy.ndarray
            The output error of the layer.

        Returns
        -------
        float
            The input error of the layer.
        """
        # computes the layer input error (the output error from the previous layer),
        # dE/dX, to pass on to the previous layer
        # SHAPES: (batch_size, input_columns) = (batch_size, output_columns) * (output_columns, input_columns)
        input_error = np.dot(output_error, self.weights.T)

        # computes the weight error: dE/dW = X.T * dE/dY
        # SHAPES: (input_columns, output_columns) = (input_columns, batch_size) * (batch_size, output_columns)
        weights_error = np.dot(self.input.T, output_error)
        # computes the bias error: dE/dB = dE/dY
        # SHAPES: (1, output_columns) = SUM over the rows of a matrix of shape (batch_size, output_columns)
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        # updates parameters
        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)
        return input_error
    
    def output_shape(self) -> tuple:
        """
        Returns the shape of the output of the layer.

        Returns
        -------
        tuple
            The shape of the output of the layer.
        """
        return (self.n_units,) 
    
class Dropout(Layer):
    """
    Dropout layer for regularization in deep neural networks.

    This layer randomly sets input units to 0 with a frequency of probability at each step 
    during training time, which helps prevent overfitting. Inputs not set to 0 are scaled up 
    by 1/(1 - probability) to maintain the sum of the inputs unchanged (Inverted Dropout).

    During inference (training=False), the layer acts as an identity function, passing 
    the input data through without any modifications.
    """

    def __init__(self, probability: float):
        """
        Initialize the Dropout layer with a specified dropout rate.

        Parameters
        ----------
        probability : float
            Probability of dropping out input units, it was to be between 0 and 1.

        Raises
        ------
        ValueError
            If the probability is not in the range (0, 1).
        """
        if not (0 <= probability <= 1):
            raise ValueError("Probability must be between 0 and 1.")
        
        self.probability = probability
        self.mask = None
        self.input = None
        self.output = None

    def forward_propagation(self, input: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Performs forward propagation on the given input, applying dropout if in training mode.

        If training is True, a mask is generated from a binomial distribution to randomly zero out 
        elements with probability. The remaining elements are scaled to maintain the expected sum of values.

        If training is False (inference mode), the input is returned.

        Parameters
        ----------
        input : np.ndarray
            The input array to the layer.
        training : bool
            Whether the layer is in training mode or inference mode. 
            - If True: Dropout mask and scaling are applied.
            - If False: Identity function (no change).

        Returns
        -------
        np.ndarray
            The output of the layer.
        """
        self.input = input

        if training:
            self.mask = np.random.binomial(1, 1 - self.probability, size=input.shape)
            scaling_factor = 1 / (1 - self.probability)
            self.output = input * self.mask * scaling_factor
            return self.output

        return input

    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        """
        Performs backward propagation on the given output error.
        Ensures that gradients are only propagated back through the neurons that were 
        active during training, while dropped neurons (mask=0) receive zero gradient.

        Parameters
        ----------
        output_error : np.ndarray
            The error gradient coming from the next layer (dE/dY).

        Returns
        -------
        np.ndarray
            The error gradient to be passed to the previous layer (dE/dX).
        """
        return output_error * self.mask

    def output_shape(self) -> tuple:
        """
        Returns the shape of the output of the layer.

        Since the Dropout layer only zeros out values without changing the dimensions 
        of the data, the output shape is identical to the input shape.

        Returns
        -------
        tuple
            The shape of the output (which is equal to the input shape).
        """
        return self.input_shape()

    def parameters(self) -> int:
        """
        Returns the number of trainable parameters in the layer.

        For the Dropout layer, this value is always 0, as it does not contain 
        any weights or biases to be updated during backpropagation. 

        Returns
        -------
        int
            The number of parameters (0).
        """
        return 0
    