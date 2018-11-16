""" Losses module for custom Keras losses."""

import numpy as np
from abc import ABC, abstractmethod
from keras import backend as K
from keras import losses


class Loss(ABC):
    """
    Probabilistic loss function class that implements keras log-
    likelihood loss and sampling from pdf for regular mean-squared error loss.
    This means that there is not much to do here as it behaves like a Gaussian
    with zero variance.

    """
    @abstractmethod
    def __call__(self):
        """The call function needs to behave like a typical Keras loss function."""
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        raise NotImplementedError


class mse(Loss):
    """Mean squared error."""

    def __init__(self, *args, **kwargs):
        """Initialize instance parameters."""
        self.keras_loss = losses.mean_squared_error

    def __call__(self, y_true, y_pred):
        """Need a call function that behaves like Keras loss."""
        return self.keras_loss(y_true, y_pred)

    def sample(self, y_pred, **kwargs):
        """No sampling happens. We just return the predicted values."""
        return y_pred


class mae(Loss):
    """Mean absolute error."""

    def __init__(self, *args, **kwargs):
        """Initialize instance parameters."""
        self.keras_loss = losses.mean_absolute_error

    def __call__(self, y_true, y_pred):
        """Need a call function that behaves like Keras loss."""
        return self.keras_loss(y_true, y_pred)

    def sample(self, y_pred, **kwargs):
        """No sampling happens. We just return the predicted values."""
        return y_pred


class gaussian(Loss):
    """
    Probabilistic loss function class that implements keras log-
    likelihood loss and sampling from pdf for a heteroscedastic Gaussian.
    :param output_shape: keras model output shape.
    :type output_shape: tuple

    """

    def __init__(self, output_shape):
        """Initialize instance parameters."""
        self.n_dim = int(output_shape[1]/2)  # Number of independent covariates

    def __call__(self, y_true, y_pred):
        """Need a call function that behaves like Keras loss."""
        # Mean and log variance are parameterized by the first
        # and second half of y_pred.
        mean = y_pred[:, :, :self.n_dim]
        log_var = y_pred[:, :, self.n_dim:]

        return self.log_likelihood(y_true, mean, log_var)

    def sample(self, y_pred, n_samples=1):
        """
        Need a sample function that takes in Keras outputs and draws
        samples from correct pdf."
        """
        # Mean and log variance are parameterized by the first
        # and second half of y_pred.
        mean = y_pred[:, :, :self.n_dim]
        log_var = y_pred[:, :, self.n_dim:]

        # Bring mean and logvar into shape
        mean = np.repeat(mean, n_samples, axis=0)
        std = np.repeat(np.exp(0.5 * log_var), n_samples, axis=0)
        size = mean.shape

        # Draw samples
        samples = np.random.standard_normal(size=size)

        return samples * std + mean

    def log_likelihood(self, y, mean, log_var):
        """Return log likelihood for this distribution."""
        # Need the invert variance parameterized by an exponential
        # for numeric stability.
        inv_var = K.exp(-log_var)

        # Calculate the mean error over all data points
        squares = K.square(mean - y)
        weighted_squares = squares * inv_var + log_var
        heteroscedastic_loss = K.mean(weighted_squares)

        return heteroscedastic_loss
