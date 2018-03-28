# -*- coding: utf-8 -*-
"""Time series regression module.

This module provides access to regressors that can be fit to univariate
or multivariate time series.

"""
import numpy as np

from abc import ABC, abstractmethod
from hyperopt import hp, fmin, tpe
from keras.optimizers import RMSprop
from deep4cast.models import TruncatedRNN


class Regressor(ABC):
    """Abstract data handler class."""

    def __init__(self):
        """Initialize properties."""
        # Model-dependent internal variables
        self._model = None
        self._history = None

        # Internal boolean checks
        self.is_fitted = False

    def fit(self, X, y, validation_split=0.0, verbose=0):
        """Fit model to data.

        :param X: Independent variable array.
        :type X: numpy.array with shape (n,m,p)
        :param y: Dependent variable array.
        :type y: numpy.array with shape (q,r)
        :param validation_split: Fraction of data for validation.
        :type validation_split: float
        :param  verbose: Toggle print to screen during fitting.
        :type verbose: int

        """
        self.check_data_format(X, y)

        # Set up the model based on internal model class which needs to be
        # provided in the concrete implementation.
        self._model = self._model_class(
            input_shape=(X.shape[1], X.shape[2]),
            topology=self._topology
        )

        # Compose and compile model with loss and optimizer, both of which
        # need to be provided in the concerte implementation.
        self._model.compile(
            loss=self._loss,
            optimizer=self._optimizer,
            metrics=self._metrics
        )

        # Fit model to data  and set internal history object for
        # ecapsulated access from outside.
        self._history = self._model.fit(
            X,
            y,
            batch_size=self._batch_size,
            epochs=self._epochs,
            verbose=verbose,
            validation_split=validation_split
        )

        self.is_fitted = True

    @abstractmethod
    def hyperfit(self, X, y, validation_split=0):
        """Optimize regressor hyperparameters and model parameters.

        :param X: Independent variable array.
        :type X: numpy.array with shape (n,m,p)
        :param y: Dependent variable array.
        :type y: numpy.array with shape (q,r)
        :param validation_split: Fraction of data for validation.
        :type validation_split: float

        """
        pass

    def predict(self, X, batch_size=None):
        """Generate predictions for input array X.

        :param X: Independent variable array.
        :type X: numpy.array with shape (n,m,p)

        """
        self.check_is_fitted()
        self.check_data_format(X, None)

        return self._model.predict(X, batch_size)

    @property
    def history(self):
        """Keras training history."""
        return self._history

    @property
    def optimizer(self):
        """Name of optimizer used for this model."""
        return self._optimizer.__name__

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @staticmethod
    def check_data_format(X, y):
        """Raise error if data has incorrect format.

        :param X: Independent variable array.
        :type X: numpy.array with shape (n,m,p)
        :param y: Dependent variable array.
        :type y: numpy.array with shape (q,r)
        :raises: Exception in case of wrong data format.

        """

        if not isinstance(X, np.ndarray) and not isinstance(y, np.ndarray):
            raise Exception('X and y should be numpy arrays.')

        if not len(X.shape) == 3 and len(y.shape) == 2:
            raise Exception('X and y should be of shape (x,y,z) and (v,w).')

    def check_is_fitted(self):
        """Raise error if model has not been fitted to data."""
        if not self.is_fitted:
            raise Exception('Model has not been fitted!')


class TruncatedRNNRegressor(Regressor):
    """Concrete implementation of Regressor using TruncatedRNN.

    Uses the TruncatedRNN model to build a regressor for time series
    forecasting..

    :param topology: Neural network topology.
    :type topology: list of integers for RNN units per layer
    :param batch_size: Number of data points in each batch.
    :type batch_size: int
    :param epochs: Number of training epochs.
    :type epochs: int
    :param lr: Learning rate for optimizer.
    :type lr: float

    """

    def __init__(self, topology=None, batch_size=10, epochs=1, lr=0.01):
        """Initialize additional properties."""
        # Hyperparemeters for this regressor class
        self._topology = topology if topology else [16]
        self._batch_size = batch_size
        self._epochs = epochs
        self._learning_rate = lr

        # Class-specific  attributes for this regressor class
        self._model_class = TruncatedRNN
        self._loss = 'mse'
        self._optimizer = RMSprop(lr=lr)
        self._metrics = ['mape']
        self._hyperoptimizer = None

    def hyperfit(self, X, y, validation_split=0):
        """Optimize regressor hyperparameters.

        :param X: Independent variable array.
        :type X: numpy.array with shape (n,m.p)
        :param y: Dependent variable array.
        :type y: numpy.array with shape (q,r)
        :param validation_split: Fraction of data for validation.
        :type validation_split: float

        """
        pass
