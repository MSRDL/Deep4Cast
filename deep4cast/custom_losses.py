""" Losses module for custom Keras losses."""

import numpy as np

from keras import backend as K
from keras import losses


class mae():
    """
    Probabilistic loss function class that implements keras log-
    likelhood loss and sampling from pdf for regular mean-absolute error loss.
    This mean, there is not much to do here as it behaves like a Laplacian
    with zero variance.
    """

    def __init__(self, **kwargs):
        """Initialize instance parameters."""
        self.dim_factor = 1

    def __call__(self, y_true, y_pred):
        """Need a call function that behaves like Keras loss."""
        return losses.mean_absolute_error(y_true, y_pred)

    def sample(self, y_pred, n_samples=1):
        """
        Need a sample function that takes in Keras outputs and draws
        samples from correct pdf."
        """
        return y_pred


class mse():
    """
    Probabilistic loss function class that implements keras log-
    likelhood loss and sampling from pdf for regular mean-squared error loss.
    This mean, there is not much to do here as it behaves like a Gaussian
    with zero variance.
    """

    def __init__(self, **kwargs):
        """Initialize instance parameters."""
        self.dim_factor = 1

    def __call__(self, y_true, y_pred):
        """Need a call function that behaves like Keras loss."""
        return losses.mean_squared_error(y_true, y_pred)

    def sample(self, y_pred, n_samples=1):
        """
        Need a sample function that takes in Keras outputs and draws
        samples from correct pdf."
        """
        return y_pred


class heteroscedastic_gaussian():
    """
    Probabilistic loss function class that implements keras log-
    likelhood loss and sampling from pdf for a heteroscedastic Gaussian.
    :param n_dim: dimensionality of the samples.
    :type n_dim: int
    """

    def __init__(self, **kwargs):
        """Initialize instance parameters."""
        self.n_dim = kwargs['n_dim']
        self.dim_factor = 2

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
        if len(y_pred.shape) == 4:
            mean = y_pred[:, :, :, :self.n_dim]
            log_var = y_pred[:, :, :, self.n_dim:]
        elif len(y_pred.shape) == 3:
            mean = y_pred[:, :, :self.n_dim]
            log_var = y_pred[:, :, self.n_dim:]
        else:
            raise(IndexError('y_pred should have 3 or 4 dimensions.'))

        return self.sample_from_likelihood(mean, log_var, n_samples=n_samples)

    def log_likelihood(self, y, mean, log_var):
        """Return log likelihood for this distribution."""
        # Need the invert variance parameterized by an exponential
        # for numeric stability.
        inv_var = K.exp(-log_var)

        # Calculate the mean error over all data points
        squares = K.square(mean - y)
        weighted_squares = 0.5 * (squares * inv_var + log_var)
        heteroscedastic_loss = K.mean(weighted_squares)

        return heteroscedastic_loss

    def sample_from_likelihood(self, mean, log_var, n_samples):
        """
        Return n_samples from the probability distribution function
        where mean and log_var are arrays of shape (n_batch, n_steps, n_vars).

        """
        if n_samples > 1:
            size = [n_samples] + list(mean.shape)
        else:
            size = mean.shape

        std = np.exp(0.5 * log_var)
        samples = np.random.normal(
            loc=mean,
            scale=std,
            size=size
        )

        return samples
