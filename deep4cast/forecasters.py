# -*- coding: utf-8 -*-
"""Time series forecasting module.

This module provides access to forecasters that can be fit to univariate
or multivariate time series given as arrays.

"""
import numpy as np

from keras.optimizers import RMSprop, SGD
from .models import LayeredTimeSeriesModel


class Forecaster():
    """General forecaster class for forecasting time series.
    :param model_class: Neural network model class.
    :type model_class: keras model
    :param optimizer: Neural network optimizer.
    :type optimizer: keras optimizer
    :param topology: Neural network topology.
    :type topology: list
    :param batch_size: Train batch size.
    :type batch_size: int
    :param epochs: Number of training epochs.
    :type epochs: int

    """

    def __init__(self, model_class, optimizer, topology, batch_size, epochs):
        """Initialize properties."""

        # Attributes related to neural network model
        self.model_class = model_class
        self.topology = topology
        self._model = None

        # Attributes related to model training
        self.optimizer = optimizer or SGD(lr=0.01)
        self.batch_size = batch_size
        self.epochs = epochs
        self.history = None
        self._loss = 'mse'
        self._metrics = ['mape']

        # Attributes related to input data (these are set during fitting)
        self.lookback_period = None
        self.data_means = None
        self.data_stds = None

        # Boolean checks
        self._is_fitted = False
        self._is_standardized = False

    def fit(self, ts, lookback_period=10):
        """Fit model to data.

        :param ts: Time series array of shape (n_steps, n_variables)
        :type ts: numpy.array
        :param lookback_period: Length of time series window for model input
        :type lookback_period: int

        """
        self.lookback_period = lookback_period
        self._check_data_format(ts)

        # When data format has been succesfully checked standardize.
        ts_standardized = self._standardize(ts)
        ts_standardized = ts_standardized.astype('float32')

        # The model expects input-output data pairs, so we create them from
        # the standardized time series arrary by windowing. Xs are 3D tensors
        # of shape number of steps * lookback_period * dimensionality and
        # ys are 2D tensors of lookback_period * dimensionality.
        X, y = self._sequentialize(ts_standardized)

        # Set up the model based on internal model class.
        self._model = self.model_class(
            input_shape=X.shape[1:],
            topology=self.topology,
        )
        self._model.compile(
            loss=self._loss,
            optimizer=self.optimizer,
            metrics=self._metrics
        )
        self.history = self._model.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=0
        )

        # Change state to fitted so that other methods work correctly.
        self._is_fitted = True

    def predict_point_estimate(self, ts):
        """Generate predictions for input time series array ts.

        :param ts: Time series array of shape (n_steps, n_variables)
        :type ts: numpy.array

        """
        self._check_is_fitted()
        self._check_is_standardized()
        self._check_data_format(ts)

        # Bring input into correct format for model train and prediction
        ts_standardized = self._standardize(ts, locked=True)
        ts_standardized = ts_standardized.astype('float32')
        X = self._sequentialize(ts_standardized)[0]

        # Undo standardization for correct scale of predicted values.
        prediction = self._model.predict(X, self.batch_size)

        return self._unstandardize(prediction)

    def _sequentialize(self, ts):
        """Sequentialize time series array."""
        # Create two numpy arrays, one for the windowed input time series X
        # and one for the corresponding output values that need to be
        # predicted.
        d = self.lookback_period
        X = [ts[i:i + d] for i in range(len(ts) - d)]
        y = [ts[i + d] for i in range(len(ts) - d)]

        return np.array(X), np.array(y)

    def _standardize(self, ts, locked=False):
        """Standardize numpy array in a NaN-friendly way."""
        # Use numpy.nanmean/nanstd to handle potential nans when
        # when standardizing data. Make sure to keep a lock on
        # internal variables during prediction step.
        if not locked:
            self.data_means = np.nanmean(ts, axis=0)
            self.data_stds = np.nanmean(ts, axis=0)
            self._is_standardized = True
        else:
            self._check_is_standardized()
        return (ts - self.data_means) / self.data_stds

    def _unstandardize(self, ts):
        """Un-standardize numpy array."""
        self._check_is_standardized()
        return ts * self.data_stds + self.data_means

    @staticmethod
    def _check_data_format(ts):
        """Raise error if data has incorrect format."""
        if not isinstance(ts, np.ndarray) or not len(ts.shape) == 2:
            raise ValueError('ts should have shape (n_steps, n_variables).')

        if np.isnan(ts).any():
            raise ValueError('ts should not contain NaNs.')

    def _check_is_fitted(self):
        """Raise error if model has not been fitted to data."""
        if not self._is_fitted:
            raise ValueError('The model has not been fitted.')

    def _check_is_standardized(self):
        """Raise error if model data was not standardized."""
        if not self._is_standardized:
            raise ValueError('The data has not been standardized.')

    def predict_samples(self, ts, n_sample=1000, quantiles=(0.025,0.975)):
        """Generate predictions for input time series array ts.
           Output mean, median, quantile predictions and prediction samples
            :param ts: Time series array of shape (n_steps, n_variables)
            :type ts: numpy.array
            :param n_sample: Number of prediction samples, at least 1
            :type n_sample: int
            :param quantiles: Tuple of quantiles to produce corresponding confidence interval
            :type quantiles: Tuple of two floats from 0.0 to 1.0, e.g. (0.025, 0.975)
        """
        self._check_is_fitted()
        self._check_is_standardized()
        self._check_data_format(ts)

        # Bring input into correct format for model train and prediction
        ts_standardized = self._standardize(ts, locked=True)
        ts_standardized = ts_standardized.astype('float32')
        X = self._sequentialize(ts_standardized)[0]

        prediction_samples = []
        for _ in range(n_sample):
            prediction = self._model.predict(X, self.batch_size)
            prediction_samples.append(self._unstandardize(prediction))
        
        prediction_samples = np.array(prediction_samples)
        lower_quantile = np.nanpercentile(prediction_samples, quantiles[0]*100, axis=0)
        upper_quantile = np.nanpercentile(prediction_samples, quantiles[1]*100, axis=0)
        median_prediction = np.nanpercentile(prediction_samples, 50.0, axis=0)
        mean_prediction = np.mean(prediction_samples, axis=0)
        return {'mean_prediction': mean_prediction, 'median_prediction': median_prediction,
                'lower_quantile': lower_quantile, 'upper_quantile': upper_quantile,
                'prediction_samples': prediction_samples}


class CNNForecaster(Forecaster):
    """Implementation of Forecaster as temporal CNN.

    :param topology: Neural network topology. 
    :type topology: list
    :param **kwargs: Hyperparameters(e.g., learning_rate, momentum, etc.).
    :type **kwargs: dict

    """

    def __init__(self, topology, **kwargs):
        """Initialize properties."""
        self.model_class = LayeredTimeSeriesModel
        self.batch_size = 10
        self.epochs = 10
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.decay = 0.1
        self.nesterov = False
        allowed_args = (
            'batch_size',
            'epochs',
            'learning_rate',
            'momentum',
            'decay'
            'nesterov'
        )
        for arg, value in kwargs.items():
            if arg in allowed_args:
                setattr(self, arg, value)
            else:
                raise ValueError('Invalid keyword argument: {}.'.format(arg))

        self.optimizer = SGD(
            lr=self.learning_rate,
            momentum=self.momentum,
            decay=self.decay,
            nesterov=self.nesterov
        )

        super().__init__(
            self.model_class,
            self.optimizer,
            topology,
            self.batch_size,
            self.epochs
        )


class RNNForecaster(Forecaster):
    """Implementation of Forecaster as truncated RNN.

    :param topology: Neural network topology.
    :type topology: dict
    :param **kwargs: Hyperparameters(e.g., learning_rate, momentum, etc.).
    :type **kwargs: dict

    """

    def __init__(self, topology, **kwargs):
        """Initialize properties."""
        self.model_class = LayeredTimeSeriesModel
        self.batch_size = 10
        self.epochs = 10
        self.learning_rate = 0.01
        allowed_args = (
            'batch_size',
            'epochs',
            'learning_rate',
        )
        for arg, value in kwargs.items():
            if arg in allowed_args:
                setattr(self, arg, value)
            else:
                raise ValueError('Invalid keyword argument: {}.'.format(arg))

        self.optimizer = RMSprop(
            lr=self.learning_rate,
        )

        super().__init__(
            self.model_class,
            self.optimizer,
            topology,
            self.batch_size,
            self.epochs
        )


if __name__ == '__main__':
    pass
