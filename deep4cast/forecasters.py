# -*- coding: utf-8 -*-
"""Time series forecasting module.

This module provides access to forecasters that can be fit multivariate time
series given as numpy arrays.

"""
from inspect import getargspec

import numpy as np
import keras.optimizers

from .models import SharedLayerModel
from .topologies import get_topology


class Forecaster():
    """General forecaster class for forecasting time series.
    :param model_class: Neural network model class.
    :type model_class: keras model
    :param optimizer: Neural network optimizer.
    :type optimizer: string
    :param topology: Neural network topology.
    :type topology: list
    :param batch_size: Training batch size.
    :type batch_size: int
    :param epochs: Number of training epochs.
    :type epochs: int
    :param uncertainty: True if applying MC Dropout after every layer.
    :type uncertainty: boolean
    :param dropout_rate: Probability of dropping a node during Dropout.
    :type dropout_rate: float

    """

    def __init__(self,
                 topology,
                 optimizer: str,
                 lag: int,
                 horizon: int,
                 batch_size: int,
                 epochs: int,
                 uncertainty=False,
                 dropout_rate=0.1,
                 **kwargs):
        """Initialize properties."""

        # Attributes related to neural network model
        self.model_class = SharedLayerModel
        self.topology = self._build_topology(topology)
        self.uncertainty = uncertainty
        self.dropout_rate = dropout_rate
        self._model = None

        # Attributes related to model training
        self.optimizer = optimizer
        self.set_optimizer_args(kwargs)

        self.lag = lag
        self.horizon = horizon
        self.batch_size = batch_size
        self.epochs = epochs
        self.history = None
        self.loss = 'mse'
        self.metrics = ['mape']
        self._tags = None
        self.seed = None

        # Attributes related to input data (these are set during fitting)
        self._data_means = None
        self._data_scales = None

        # Boolean checks
        self._is_fitted = False
        self._is_standardized = False

    def fit(self, data, tags=None, verbose=0):
        """Fit model to data.

        :param data: Time series array of shape (n_steps, n_variables).
        :type data: numpy.array
        :param tags: Dict that contains the indices of targets and covariates.
        :type tags: dict
        :param verbose: Verbosity mode. 0 = silent, 1 = progress bar,
            2 = one line per epoch.
        :type verbose: int

        """
        self._check_data_format(data)

        # We sometimes want to fix the pseudo random number generator seed
        # when we are debugging.
        if self.seed:
            np.random.seed(self.seed)

        # Store a dictionary of lists that contain the indicies of
        # target variables, dynamic, and static covariates.
        self._tags = tags

        # When data format has been succesfully checked standardize it.
        data_standardized = self._standardize(data)
        data_standardized = data_standardized.astype('float32')

        # The model expects input-output data pairs, so we create them from
        # the standardized time series arrary by windowing. Xs are 3D tensors
        # of shape number of steps * lag * dimensionality and
        # ys are 2D tensors of lag * dimensionality.
        X, y = self._sequentialize(data_standardized)

        # Set up the model based on internal model class.
        self._model = self.model_class(
            input_shape=X.shape[1:],
            output_shape=y.shape[1:],
            topology=self.topology,
            uncertainty=self.uncertainty,
            dropout_rate=self.dropout_rate
        )

        # Keras needs to compile the comuptational graph before fitting.
        self._model.compile(
            loss=self.loss,
            optimizer=self._optimizer,
            metrics=self.metrics
        )

        # Print the model topology and parameters before fitting.
        self.history = self._model.fit(
            X,
            y,
            batch_size=int(self.batch_size),
            epochs=int(self.epochs),
            verbose=verbose
        )

        # Change state to fitted so that other methods work correctly.
        self._is_fitted = True
        self.summary = self._model.summary

    def predict(self, data, n_samples=100, quantiles=(5, 95)):
        """Generate predictions for input time series numpy array.

        :param data: Time series array of shape (n_steps, n_variables).
        :type data: numpy.array
        :param n_samples: Number of prediction samples (>= 1).
        :type n_samples: int
        :param quantiles: Tuple of quantiles for credicble interval.
        :type quantiles: tuple

        """
        n_series = data.shape[2]  # number of distinct time series to train on

        self._check_is_fitted()
        self._check_is_standardized()
        self._check_data_format(data)

        # Bring input into correct format for model train and prediction
        data_standardized = self._standardize(data, locked=True)
        data_standardized = data_standardized.astype('float32')
        X = self._sequentialize(data_standardized)[0]  # Only get inputs

        # If uncertainty is False, only do one sample prediction
        if not self.uncertainty:
            n_samples = 1

        # Repeat the prediction n_samples times to generate samples from
        # approximate posterior predictive distribution.
        samples = []
        for _ in range(n_samples):
            prediction = []
            raw_prediction = self._model.predict(X, self.batch_size)

            # We need to reashape the raw_predictios in case the forecasting
            # horizon is larger than 1 and in case more than one time series
            # is fed into the model.The Keras model has no
            # intrinsic notion of horizon.s
            stride = int(len(raw_prediction) / n_series)  # reshaping stride
            for i in range(n_series):
                # Extract the prediciton for each individual series
                prediction.append(
                    raw_prediction[i * stride:(i + 1) * stride].T
                )
            prediction = np.array(prediction)

            # Reshape according to horizon length. Needs to happen to undo
            # standadization.
            if self.horizon == 1:
                prediction = np.swapaxes(prediction, 0, 2)
            else:
                prediction = np.swapaxes(prediction, 0, 3)
                prediction = np.swapaxes(prediction, 1, 2)

            samples.append(self._unstandardize(prediction))

        samples = np.array(samples)

        # Calculate mean prediction.
        mean_prediction = np.mean(samples, axis=0)

        # Turn samples into quantiles for easier display later.
        lower_quantile = np.nanpercentile(
            samples, quantiles[0], axis=0) if self.uncertainty else None
        upper_quantile = np.nanpercentile(
            samples, quantiles[1], axis=0) if self.uncertainty else None
        median_prediction = np.nanpercentile(
            samples, 50, axis=0) if self.uncertainty else None

        return {'mean': mean_prediction,
                'median': median_prediction,
                'lower_quantile': lower_quantile,
                'upper_quantile': upper_quantile,
                'samples': samples}

    @property
    def horizon(self):
        """Return the horizon."""
        return self._horizon

    @horizon.setter
    def horizon(self, horizon):
        """Instantiate the horizon."""
        self._horizon = int(horizon)

    @property
    def lag(self):
        """Return the lag."""
        return self._lag

    @lag.setter
    def lag(self, lag):
        """Instantiate the lag."""
        self._lag = int(lag)

    @property
    def batch_size(self):
        """Return the batch_size."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        """Instantiate the batch_size."""
        self._batch_size = int(batch_size)

    @property
    def epochs(self):
        """Return the epochs."""
        return self._epochs

    @epochs.setter
    def epochs(self, epochs):
        """Instantiate the epochs."""
        self._epochs = int(epochs)

    @property
    def optimizer(self):
        """Return the optimizer name."""
        return self._optimizer.__class__.__name__

    @optimizer.setter
    def optimizer(self, optimizer):
        """Instantiate the optimizer."""
        optimizer_class = getattr(keras.optimizers, optimizer)
        self._optimizer = optimizer_class()

    def set_optimizer_args(self, params):
        """Set optimizer attributes."""
        optimizer_class = self._optimizer.__class__
        optimizer_args = getargspec(optimizer_class)[0]
        for key, value in params.items():
            if key in optimizer_args:
                setattr(self._optimizer, key, value)

    @staticmethod
    def _build_topology(topology):
        """Return topology depending on user input (str or list)."""
        if isinstance(topology, str):
            return get_topology(topology)
        if isinstance(topology, list):
            return topology

    def _sequentialize(self, data):
        """Sequentialize time series array.
        Create two numpy arrays, one for the windowed input time series X
        and one for the corresponding output values that need to be
        predicted.
        """
        n_time_steps = data.shape[0]
        n_series = data.shape[2]  # number of distinct time series to train on
        horizon = self.horizon  # Redefine variable to keep visual noise low
        lag = self.lag  # Redefine variable to keep visual noise low
        tags = self._tags

        # Sequentialize the dataset, i.e., split it into shorter windowed
        # sequences.
        X, y = [], []
        for i in range(n_series):
            for j in range(n_time_steps - lag):
                if j + lag + horizon <= n_time_steps:
                    X.append(data[j:j + lag, :, i])

                    # Target indices for forecasting
                    target_inds = range(j + lag, j + lag + horizon)
                    if tags:
                        y.append(data[target_inds, tags['targets'], i])
                    else:
                        y.append(np.squeeze(data[target_inds, :, i]))

        # Make sure to return numpy arrays not lists.
        if not X or not y:
            raise ValueError(
                'Time series is too short for lag and/or horizon. lag {} + horizon {} > n_time_steps {}.'.format(
                    lag, horizon,
                    n_time_steps
                )
            )
        return np.array(X), np.array(y)

    def _standardize(self, data, locked=False):
        """Standardize numpy array in a NaN-friendly way.

        :param data: Input time series.
        :type data: numpy array
        :param data: Boolean that locks down the scales of the data.
        :type data: boolean

        """
        # Use numpy.nanmean/nanstd to handle potential nans when
        # when standardizing data. Make sure to keep a lock on
        # internal variables during prediction step.
        if not locked:
            # By default the data scale is set by the standard deviation,
            # but we need to handle the case where the standard deviation
            # or the mean is zero.
            means = np.nanmean(data, axis=0)
            scales = np.nanstd(data, axis=0)

            # Check for small scales in stds.
            ind = np.where(scales < 1e-16)
            scales[ind] = means[ind]

            # Check again for means.
            ind = np.where(scales < 1e-16)
            scales[ind] = 1.0

            self._data_means = means
            self._data_scales = scales
            self._is_standardized = True
        else:
            self._check_is_standardized()

        return (data - self._data_means) / self._data_scales

    def _unstandardize(self, data):
        """Un-standardize numpy array."""
        self._check_is_standardized()

        # Unstandardize taking presence of covariates into account.
        if self._tags:
            means = self._data_means[self._tags['targets']]
            scales = self._data_scales[self._tags['targets']]
            return data * scales + means
        else:
            return data * self._data_scales + self._data_means

    def _check_data_format(self, data, tags=None):
        """Raise error if data has incorrect format."""
        if not isinstance(data, np.ndarray) or not len(data.shape) == 3:
            raise ValueError('data shape != (n_steps, n_vars, n_series).')

        # Check if data has any NaNs.
        if np.isnan(data).any():
            raise ValueError('data should not contain NaNs.')

        # Check if data is long enough for forecasting horizon.
        if len(data) <= self.horizon:
            raise ValueError('Time series must be longer than horizon.')

        # Make sure tags has the right format.
        if tags and not isinstance(tags, dict):
            raise ValueError('tags need to be a dictionary')

    def _check_is_fitted(self):
        """Raise error if model has not been fitted to data."""
        if not self._is_fitted:
            raise ValueError('The model has not been fitted.')

    def _check_is_standardized(self):
        """Raise error if model data was not standardized."""
        if not self._is_standardized:
            raise ValueError('The data has not been standardized.')
