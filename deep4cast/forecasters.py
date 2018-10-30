"""Time series forecasting module.
This module provides access to forecasters that can be fit multivariate time
series given as numpy arrays.
"""
from inspect import getargspec
from typing import Union
import numpy as np
import keras.optimizers

from keras.callbacks import TerminateOnNaN
from . import loss_functions

class Forecaster():
    """Forecaster class.

    Take a keras model and builds a forecaster.

    """

    def __init__(self,
                 model: Union[str, keras.Model],
                 loss: Union[str, loss_functions.Loss]='gaussian',
                 optimizer='adam',
                 batch_size=16,
                 epochs=100,
                 **kwargs):
        """Initialize properties."""
        # Neural network model attributes
        # REVIEW Toby: Can't model be a string, similar to loss?
        self.model = model

        # Optimizer attributes
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.history = None
        self._loss = loss

        # Configure optimizer
        allowed_kwargs = self.get_optimizer_args()
        for key, value in kwargs.items():
            if key not in allowed_kwargs:
                raise ValueError('{} not a valid argument.'.format(key))
        self.set_optimizer_args(kwargs)

        # Boolean checks for refitting
        self._is_fitted = False

    def fit(self, X, y, verbose=0):
        """Fit model to data."""
        # Make sure the data type is float32 for optimal performance of all
        # Keras backends.
        X = X.astype('float32')
        y = y.astype('float32')

        # TODO: all initializations should happen in __init__
        # Users should be mindful of the input and output dimensions.
        if not isinstance(self._loss, loss_functions.Loss):
            self._loss = getattr(loss_functions, self._loss)(n_dim=y.shape[2])

        loss_dim_factor = self._loss.dim_factor
        output_shape = (y.shape[1], y.shape[2] * loss_dim_factor)

        # Need to handle the case where the model is fitted for more epochs
        # after it has already been fitted
        if not self._is_fitted:
            # TODO: all initializations should happen in __init__
            # Users should be mindful of the input and output dimensions. Plus, they may want to design their own net, which shouldn't
            # require implementing `build_layers`.
            self.model.build_layers(
                input_shape=X.shape[1:],
                output_shape=output_shape
            )

            # Keras needs to compile the computational graph before fitting
            self.model.compile(loss=self._loss, optimizer=self._optimizer)

        # Fit model to data
        self.history = self.model.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[TerminateOnNaN()],
            verbose=verbose,
        )

        # Change state to fitted so that other methods work correctly
        self._is_fitted = True

    def predict(self, X, n_samples=1000):
        """Generate predictions for input time series numpy array.
        :param data: Time series array of shape (n_steps, n_variables).
        :type data: numpy.array
        :param n_samples: Number of prediction samples (>= 1).
        :type n_samples: int

        """
        # Check if model is actually fitted
        self.check_is_fitted()

        # Make sure the data type is float32
        X = X.astype('float32')

        # Repeat the prediction n_samples times to generate samples from
        # approximate posterior predictive distribution.
        block_size = len(X)
        X = np.repeat(X, [n_samples] * len(X), axis=0)

        # Make predictions for parameters of pdfs then sample from pdfs
        predictions = self.model.predict(X, self.batch_size)
        predictions = self._loss.sample(predictions, n_samples=1)

        # Reorganize prediction samples into 3D array
        reshuffled_predictions = []
        for i in range(block_size):
            block = predictions[i * n_samples:(i + 1) * n_samples]
            block = np.expand_dims(block, axis=1)
            reshuffled_predictions.append(block)
        predictions = np.concatenate(reshuffled_predictions, axis=1)

        return predictions

    def check_is_fitted(self):
        """Raise error if model has not been fitted to data."""
        if not self._is_fitted:
            raise ValueError('The model has not been fitted.')

    def reset(self):
        """Reset model for refitting."""
        self._is_fitted = False

    @property
    def optimizer(self):
        """Return the optimizer name only."""
        return self._optimizer.__class__.__name__

    @optimizer.setter
    def optimizer(self, optimizer):
        """Instantiate the optimizer."""
        optimizer_class = getattr(keras.optimizers, optimizer)
        self._optimizer = optimizer_class()

    def get_optimizer_args(self):
        """Get optimizer parameters."""
        args = getargspec(self._optimizer.__class__)[0]  # REVIEW @Toby: getargspec is being deprecated
        args.remove('self')
        return args

    def set_optimizer_args(self, params):
        """Set optimizer parameters."""
        optimizer_class = self._optimizer.__class__
        optimizer_args = getargspec(optimizer_class)[0]
        for key, value in params.items():
            if key in optimizer_args:
                setattr(self._optimizer, key, value)


class VectorScaler():
    """Scaler class.

    Rescale vectors removing mean and dividing by standard deviation
    on a component basis.

    :param targets: targets in the data that should be rescaled.
    :type targets: list

    """

    def __init__(self, targets=None):
        """Initialize properties."""
        self.targets = targets
        self.x_mean = None
        self.x_std = None
        self.x_is_fitted = False
        self.y_mean = None
        self.y_std = None
        self.y_is_fitted = False

    def fit_x(self, X):
        """Fit the scaler."""
        if self.targets is None:
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
        else:
            # Need to concatenate mean with zeros and stds with ones for
            # categorical targets
            mean = np.zeros(X.shape[1:])
            std = np.ones(X.shape[1:])
            mean[:, self.targets] = np.mean(X[:, :, self.targets], axis=0)
            std[:, self.targets] = np.std(X[:, :, self.targets], axis=0)

        self.x_mean = mean
        self.x_std = std
        self.x_is_fitted = True

    def fit_y(self, y):
        """Fit the scaler."""
        self.y_mean = np.mean(y, axis=0)
        self.y_std = np.std(y, axis=0)
        self.y_is_fitted = True

    def transform_x(self, X):
        return (X - self.x_mean) / self.x_std

    def transform_y(self, y):
        return (y - self.y_mean) / self.y_std

    def fit_transform_x(self, X):
        self.fit_x(X)
        return self.transform_x(X)

    def fit_transform_y(self, y):
        self.fit_y(y)
        return self.transform_y(y)

    def inverse_transform_x(self, X):
        if self.x_is_fitted:
            return X * self.x_std + self.x_mean
        else:
            raise ValueError('Not fitted on X.')

    def inverse_transform_y(self, y):
        if self.y_is_fitted:
            return y * self.y_std + self.y_mean
        else:
            raise ValueError('Not fitted on y.')
