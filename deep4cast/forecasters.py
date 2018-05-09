# -*- coding: utf-8 -*-
"""Time series forecasting module.

This module provides access to forecasters that can be fit to univariate
or multivariate time series given as arrays.

"""

import numpy as np

from keras.optimizers import RMSprop, SGD
from .models import SharedLayerModel


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
    :param uncertainty: What type of uncertainty. None = ignore uncertainty,
        'all' = add dropout to every layer, 'last' = add dropout before 
        output node. #ToDo: 'last' is not generic and doesn't sensible either.
    :type uncertainty: string
    :param dropout_rate:  Fraction of the units to drop for the linear
        transformation of the inputs. Float between 0 and 1.
    :type dropout_rate: float

    """

    def __init__(self,
                 model_class,
                 optimizer,
                 topology,
                 batch_size,
                 epochs: int,
                 uncertainty: str, #ToDo: indicate the parameter type here, not in docstring
                 dropout_rate):
        """Initialize properties."""

        # Attributes related to neural network model
        self.model_class = model_class
        self.topology = topology
        self.uncertainty = uncertainty
        self.dropout_rate = dropout_rate
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


    # REVIEW: the more commonly used term for `lookback_period` is `lag`.
    def fit(self, ts, lookback_period=10, verbose=2):
        """Fit model to data.

        :param ts: Time series array of shape (n_steps, n_variables)
        :type ts: numpy.array
        :param lookback_period: Length of time series window for model input
        :type lookback_period: int
        :param verbose: Verbosity mode. 0 = silent, 1 = progress bar,
            2 = one line per epoch.
        :type verbose: int

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
            uncertainty=self.uncertainty,
            dropout_rate=self.dropout_rate
        )
        self._model.compile(
            loss=self._loss,
            optimizer=self.optimizer,
            metrics=self._metrics
        )
        print(self._model.summary())
        self.history = self._model.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=verbose
        )

        # Change state to fitted so that other methods work correctly.
        self._is_fitted = True

    def predict(self, ts, n_sample=1000, quantiles=(0.025, 0.975)):
        """Generate predictions for input time series array ts.
           Output mean, median, quantile predictions and prediction samples
            :param ts: Time series array of shape (n_steps, n_variables)
            :type ts: numpy.array
            :param n_sample: Number of prediction samples, at least 1
            :type n_sample: int
            :param quantiles: Tuple of quantiles to produce corresponding
                confidence interval
            :type quantiles: Tuple of two floats from 0.0 to 1.0,
                e.g. (0.025, 0.975)
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
        lower_quantile = np.nanpercentile(
            prediction_samples, quantiles[0]*100, axis=0)
        upper_quantile = np.nanpercentile(
            prediction_samples, quantiles[1]*100, axis=0)
        median_prediction = np.nanpercentile(prediction_samples, 50.0, axis=0)
        mean_prediction = np.mean(prediction_samples, axis=0)
        return {'mean_prediction': mean_prediction,
                'median_prediction': median_prediction,
                'lower_quantile': lower_quantile,
                'upper_quantile': upper_quantile,
                'prediction_samples': prediction_samples}

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


# Review: simplify the code by having just one Forecaster class. The network topology can be passed in as an argument.
# We can have a few pre-defined networks such as RNN, CNN, LSTNet, etc.from
# Example command-line usage would be:
#   python deep4cast.py --data-path data.csv --network rnn ...
class CNNForecaster(Forecaster):
    """Implementation of Forecaster as temporal CNN.

    :param topology: Neural network topology.
    :type topology: list
    :param **kwargs: Hyperparameters(e.g., learning_rate, momentum, etc.).
    :type **kwargs: dict

    """

    def __init__(self, topology, **kwargs):
        """Initialize properties."""
        self.model_class = SharedLayerModel
        self.batch_size = 10
        self.epochs = 10
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.decay = 0.1
        self.nesterov = False
        self.uncertainty = None
        self.dropout_rate = 0.1
        allowed_args = (
            'batch_size',
            'epochs',
            'learning_rate',
            'momentum',
            'decay',
            'nesterov',
            'uncertainty',
            'dropout_rate'
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
            self.epochs,
            self.uncertainty,
            self.dropout_rate
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
        self.model_class = SharedLayerModel
        self.batch_size = 10
        self.epochs = 10
        self.learning_rate = 0.01
        self.uncertainty = None
        self.dropout_rate = 0.1
        allowed_args = (
            'batch_size',
            'epochs',
            'learning_rate',
            'uncertainty',
            'dropout_rate'
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
            self.epochs,
            self.uncertainty,
            self.dropout_rate
        )


if __name__ == '__main__':
    pass
