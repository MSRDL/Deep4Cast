# -*- coding: utf-8 -*-
"""Time series forecasting module.

This module provides access to forecasters that can be fit to univariate
or multivariate time series given as arrays.

"""
import numpy as np

from keras.optimizers import RMSprop, SGD
from .models import LayeredTimeSeriesModel


class Forecaster(object):
    """General forecaster class for forecasting time series.

    :param topology: Neural network topology.
    :type topology: dict
    :param optimizer: Loss optimizer.
    :type optimizer: string
    :param **kwargs: Hyperparameters(e.g., learning_rate, momentum, etc.).
    :type **kwargs: dict

    """

    def __init__(self, topology=None, optimizer=None, **kwargs):
        """Initialize properties."""
        # Hyperparameters
        self._topology = topology or []
        self._batch_size = None
        self._epochs = 50
        self._learning_rate = 0.01
        self._momentum = None
        self._decay = None
        for key, value in kwargs.items():
            if key in (
                'batch_size',
                'epochs',
                'learning_rate',
                'momentum',
                'decay'
            ):
                setattr(self, '_' + key, value)

        # Optimizer-related attributes
        self._optimizer = optimizer or SGD(lr=self._learning_rate)
        self._loss = 'mse'
        self._metrics = ['mape']
        self._model = None
        self._history = None

        # Data-related attributes
        self._lookback_period = None
        self._data_mean = None
        self._data_std = None

        # Internal boolean checks
        self._is_fitted = False
        self._is_standardized = False

    def fit(self, T, lookback_period=10):
        """Fit model to data.

        :param T: Time series array of shape (n_steps, n_variables)
        :type T: numpy.array
        :param lookback_period: Length of time series window for model input
        :type lookback_period: int

        """
        self._lookback_period = lookback_period
        self._check_data_format(T)

        # When data format has been succesfully checked, we can safely
        # standardize.
        _T = self._standardize(T)

        # The model expects input-output data pairs, so we create them from
        # the standardized time series arrary by windowing. Xs are 3D tensors
        # of shape number of steps * lookback_period * dimensionality and
        # ys are 2D tensors of lookback_period * dimensionality.
        _X, _y = self._sequentialize(_T)

        # Set up the model based on internal model class.
        # The data input shape to the Keras model should be
        # steps * lookback_period * dimensionality.
        self._model = LayeredTimeSeriesModel(
            input_shape=_X.shape[1:],
            topology=self._topology,
        )

        # Compose and compile model with loss and optimizer.
        self._model.compile(
            loss=self._loss,
            optimizer=self._optimizer,
            metrics=self._metrics
        )

        # Fit model to data and set internal history object for
        # encapsulated access from outside.
        self._history = self._model.fit(
            _X,
            _y,
            batch_size=self._batch_size,
            epochs=self._epochs,
            verbose=0
        )

        # Change state to fitted so that other methods work correctly.
        self._is_fitted = True

    def predict(self, T):
        """Generate predictions for input time series array T.

        :param T: Time series array of shape (n_steps, n_variables)
        :type T: numpy.array

        """
        self._check_is_fitted()
        self._check_is_standardized()
        self._check_data_format(T)

        # Bring input into correct format for model prediction method
        _T = self._standardize(T, locked=True)
        _X = self._sequentialize(_T)[0]

        # Undo standardization after prediction for correct scale of
        # predicted values.
        prediction = self._model.predict(_X, self._batch_size)
        prediction = self._unstandardize(prediction)
        return prediction

    @property
    def history(self):
        """Keras training history property."""
        return self._history

    def _sequentialize(self, T):
        """Sequentialize time series array."""
        # Create two numpy arrays, one for the windowed input time series _X
        # and one for the corresponding output values that need to be
        # predicted.
        d = self._lookback_period
        X = [T[i:i + d] for i in range(len(T) - d)]
        y = [T[i + d] for i in range(len(T) - d)]
        return np.array(X), np.array(y)

    def _standardize(self, T, locked=False):
        """Standardize numpy array in a NaN-friendly way."""
        # Use numpy.nanmean/nanstd to handle potential nans when
        # when standardizing data. Make sure to keep a lock on
        # internal variables during prediction step.
        if not locked:
            self._means = np.nanmean(T, axis=0)
            self._stds = np.nanmean(T, axis=0)
            self._is_standardized = True
        else:
            self._check_is_standardized()
        return (T - self._means) / self._stds

    def _unstandardize(self, T):
        """Un-standardize numpy array."""
        self._check_is_standardized()
        return T * self._stds + self._means

    @staticmethod
    def _check_data_format(T):
        """Raise error if data has incorrect format."""
        if not isinstance(T, np.ndarray) or not len(T.shape) == 2:
            raise ValueError('T should have shape (n_steps, n_variables).')

        if np.isnan(T).any():
            raise ValueError('T should not contain NaNs.')

    def _check_is_fitted(self):
        """Raise error if model has not been fitted to data."""
        if not self._is_fitted:
            raise ValueError('The model has not been fitted.')

    def _check_is_standardized(self):
        """Raise error if model data was not standardized."""
        if not self._is_standardized:
            raise ValueError('The data has not been standardized.')


class CNNForecaster(Forecaster):
    """Implementation of Forecaster as temporal CNN.

    :param topology: Neural network topology. If None, simple CNN is used.
    :type topology: dict
    :param **kwargs: Hyperparameters(e.g., learning_rate, momentum, etc.).
    :type **kwargs: dict

    """

    def __init__(self, topology=None, **kwargs):
        """Initialize properties."""
        # Hyperparameters
        if topology:
            super(CNNForecaster, self).__init__(topology, **kwargs)
        else:
            super(CNNForecaster, self).__init__(
                [
                    ('Conv1D', {'filters': 64,
                                'kernel_size': 5,
                                'activation': 'elu'}),
                    ('MaxPooling1D', {'pool_size': 3,
                                      'strides': 1}),
                    ('Flatten', {}),
                    ('Dense', {'units': 64})
                ],
                **kwargs
            )
        self._learning_rate = 0.1
        self._momentum = 0.9
        self._decay = 0.1

        # Optimizer-related attributes
        self._optimizer = SGD(
            lr=self._learning_rate,
            momentum=self._momentum,
            decay=self._decay,
            nesterov=False
        )


class RNNForecaster(Forecaster):
    """Implementation of Forecaster as truncated RNN.

    :param topology: Neural network topology. If None, simple CNN is used.
    :type topology: dict
    :param **kwargs: Hyperparameters(e.g., learning_rate, momentum, etc.).
    :type **kwargs: dict

    """

    def __init__(self, topology=None, **kwargs):
        """Initialize properties."""
        # Hyperparameters
        if topology:
            super(RNNForecaster, self).__init__(topology, **kwargs)
        else:
            super(RNNForecaster, self).__init__(
                [('GRU', {'units': 32})],
                **kwargs
            )
        self._learning_rate = 0.01

        # Optimizer-related attributes
        self._optimizer = RMSprop(lr=self._learning_rate)


if __name__ == '__main__':
    pass
