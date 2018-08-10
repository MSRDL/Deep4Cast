# -*- coding: utf-8 -*-
"""Time series forecasting module.
This module provides access to forecasters that can be fit multivariate time
series given as numpy arrays.
"""
from inspect import getargspec

import time
import numpy as np
import pandas as pd
import keras.optimizers
from keras.callbacks import EarlyStopping

from . import custom_losses, custom_metrics


class Forecaster():
    """General forecaster class for forecasting time series.
    :param model_class: Neural network model class.
    :type model_class: keras model
    :param optimizer: Neural network optimizer.
    :type optimizer: string
    :param topology: Neural network topology.
    :type topology: keras.model
    :param batch_size: Training batch size.
    :type batch_size: int
    :param max_epochs: Maximum number of training max_epochs.
    :type max_epochs: int
    :param val_frac: Fraction of data points to set aside for validation.
    :type val_frac: float
    :param patience: number of max_epochs to wait before early stopping,
    :type patience: float

    """

    def __init__(self,
                 topology,
                 lag,
                 horizon,
                 loss='heteroscedastic_gaussian',
                 optimizer='adam',
                 batch_size=16,
                 max_epochs=100,
                 val_frac=0.1,
                 patience=5,
                 **kwargs):
        """Initialize properties."""

        # Neural network model attributes
        self.model = topology

        # Model training attributes
        self.optimizer = optimizer
        self.set_optimizer_args(kwargs)

        self.lag = lag
        self.horizon = horizon
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.val_frac = val_frac
        self.patience = patience
        self.loss = loss
        self._loss = None
        self.history = None
        self.targets = None

        # Data attributes (these are set during fitting)
        self._features_means = None
        self._features_scales = None
        self._targets_means = None
        self._targets_scales = None

        # Boolean checks
        self._is_fitted = False
        self._is_normalized = False

    def fit(self, data, targets=None, verbose=0):
        """Fit model to data.
        :param data: Time series array of shape (n_steps, n_variables).
        :type data: numpy.array
        :param targets: List of covariates indices.
        :type targets: list
        :param verbose: Verbosity mode. 0 = silent, 1 = progress bar,
            2 = one line per epoch.
        :type verbose: int
        """
        self.targets = targets

        # Check if data doesn't contains NaNs and such
        self._check_data_format(data)

        # Make sure the floats have the correct format
        data = self._convert_to_float32(data)

        # Split the input data into training and validation set to handle
        # early stopping
        val_length = int(max([len(x) for x in data]) * self.val_frac)
        data_train, data_val = [], []
        for time_series in data:
            data_train.append(time_series[:-val_length, :])
            data_val.append(time_series[-val_length-self.lag:, :])
        data_train = np.array(data_train)
        data_val = np.array(data_val)

        # The model expects input-output data pairs, so we create them from
        # the time series arrary by windowing. Xs are 3D tensors
        # of shape number of batch_size, timesteps, input_dim and
        # ys are 2D tensors of lag * dimensionality.
        X_train, y_train = self._sequentialize(data_train)
        X_val, y_val = self._sequentialize(data_val)

        # Remove NaN's that occur during windowing
        X_train = X_train[~np.isnan(y_train)[:, 0, 0]]
        y_train = y_train[~np.isnan(y_train)[:, 0, 0]]
        X_val = X_val[~np.isnan(y_val)[:, 0, 0]]
        y_val = y_val[~np.isnan(y_val)[:, 0, 0]]

        # Prepare model output shape based on loss function type to handle
        # loss functions that requires multiple parameters such as the
        # heteroscedsatic Gaussian
        if isinstance(self.loss, str):
            self._loss = getattr(custom_losses, self.loss)(
                n_dim=y_train.shape[2]
            )
        else:
            self._loss = self.loss
        loss_dim_factor = self._loss.dim_factor
        output_shape = (y_train.shape[1], y_train.shape[2] * loss_dim_factor)

        # Need to handle the case where the model is fitted for more epochs
        # after it has already been fitted
        if not self._is_fitted:
            # Set up the model based on internal model class
            self.model.build_layers(
                input_shape=X_train.shape[1:],
                output_shape=output_shape
            )

            # Keras needs to compile the computational graph before fitting
            self.model.compile(
                loss=self._loss,
                optimizer=self._optimizer
            )

        # Built-in early stopping as Keras callback
        es = EarlyStopping(monitor='val_loss', patience=self.patience)

        # Fit model to data
        self.history = self.model.fit(
            X_train,
            y_train,
            shuffle=True,
            batch_size=int(self.batch_size),
            epochs=int(self.max_epochs),
            verbose=verbose,
            validation_data=(X_val, y_val),
            callbacks=[es]
        )

        # Change state to fitted so that other methods work correctly
        self._is_fitted = True

    def predict(self, data, n_samples=1000, quantiles=(5, 95)):
        """Generate predictions for input time series numpy array.
        :param data: Time series array of shape (n_steps, n_variables).
        :type data: numpy.array
        :param n_samples: Number of prediction samples (>= 1).
        :type n_samples: int
        :param quantiles: Tuple of quantiles for credicble interval.
        :type quantiles: tuple
        """
        # Check if model is actually fitted
        self._check_is_fitted()

        # Now only use the last window from the input sequences to predict as
        # that is the only part of the input data that is needed for
        # prediction
        data_pred = []
        for time_series in data:
            data_pred.append(time_series[-self.lag:, :])
        data_pred = np.array(data_pred)

        # Make sure the floats have the correct format
        data_pred = self._convert_to_float32(data_pred)

        means, samples = [], []
        for time_series in data_pred:
            time_series = np.expand_dims(time_series, 0)
            # The model expects input-output data pairs, so we create them from
            # the time series arrary by windowing. Xs are 3D tensors
            # of shape number of batch_size, timesteps, input_dim and
            # ys are 2D tensors of lag * dimensionality.
            X, __ = self._sequentialize(time_series)

            # Repeat the prediction n_samples times to generate samples from
            # approximate posterior predictive distribution.
            block_size = len(X)
            X = np.repeat(X, [n_samples for _ in range(len(X))], axis=0)

            # Make predictions for parameters of pdfs then sample from pdfs
            predictions = self.model.predict(X, self.batch_size)
            predictions = self._loss.sample(
                predictions,
                n_samples=10
            )

            # Calculate the mean of the predictions
            reshuffled_predictions = []
            for i in range(n_samples):
                block = predictions[i * block_size:(i + 1) * block_size]
                reshuffled_predictions.append(block)
            predictions = np.array(reshuffled_predictions)
            mean = np.mean(predictions, axis=0)
            means.append(mean)
            samples.append(predictions)

        means = np.array(means)[:, 0, :, :]
        samples = np.array(samples)[:, :, 0, :]
        samples = np.swapaxes(samples, 0, 1)

        return {'mean': means,
                'samples': samples}

    @staticmethod
    def _convert_to_float32(array):
        """Converts all time series in an array to float32."""
        out_array = np.copy(array)
        for i, sub_array in enumerate(array):
            out_array[i] = sub_array.astype('float32')
        return out_array

    def _sequentialize(self, data):
        """Sequentialize time series array.
        Create two numpy arrays, one for the windowed input time series X
        and one for the corresponding output values that need to be
        predicted.
        """
        # Redefine variable to keep further visual noise in code lower
        horizon = self.horizon
        lag = self.lag

        # Sequentialize the dataset, i.e., split it into shorter windowed
        # sequences.
        X, y = [], []
        for time_series in data:
            # Making sure the time_series dataset is in correct format
            time_series = np.atleast_2d(time_series)

            # Need the number of time steps per window and the number of
            # covariates
            n_time_steps, n_vars = time_series.shape

            # No build the data structure
            for j in range(n_time_steps - lag + 1):
                lag_ts = time_series[j:j + lag]
                forecast_ts = time_series[j + lag:j + lag + horizon]
                if len(forecast_ts) < horizon:
                    forecast_ts = np.ones(shape=(horizon, n_vars)) * np.nan
                X.append(lag_ts)
                if self.targets:
                    y.append(forecast_ts[:, self.targets])
                else:
                    y.append(forecast_ts)

        if not X or not y:
            raise ValueError(
                'Time series is too short for lag and/or horizon. lag {} + horizon {} > n_time_steps {}.'.format(
                    lag, horizon,
                    n_time_steps
                )
            )

        # Make sure we output numpy arrays.
        X = np.array(X)
        y = np.array(y)
        return X, y

    def _check_data_format(self, data):
        """Raise error if data has incorrect format."""
        # Check if data has any NaNs.
        if np.isnan([np.isnan(x).any() for x in data]).any():
            raise ValueError('data should not contain NaNs.')

        # Check if data is long enough for forecasting horizon.
        if np.array([len(x) <= self.horizon for x in data]).any():
            raise ValueError('Time series must be longer than horizon.')

    def _check_is_fitted(self):
        """Raise error if model has not been fitted to data."""
        if not self._is_fitted:
            raise ValueError('The model has not been fitted.')

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
    def max_epochs(self):
        """Return the max_epochs."""
        return self._max_epochs

    @max_epochs.setter
    def max_epochs(self, max_epochs):
        """Instantiate the max_epochs."""
        self._max_epochs = int(max_epochs)

    @property
    def patience(self):
        """Return the patience."""
        return self._patience

    @patience.setter
    def patience(self, patience):
        """Instantiate the patience."""
        self._patience = int(patience)

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


class CrossValidator():
    """Temporal cross-validator class.

    This class performs temporal (causal) cross-validation similar to the
    approach in https://robjhyndman.com/papers/cv-wp.pdf.

    :param forecaster: Forecaster.
    :type forecaster: A forecaster class
    :param val_frac: Fraction of data to be used for validation per fold.
    :type val_frac: float
    :param n_folds: Number of temporal folds.
    :type n_folds: int
    :param loss: The kind of loss used for evaluating the forecaster on folds.
    :type loss: string

    """

    def __init__(self,
                 forecaster: Forecaster,
                 test_fraction=0.1,
                 n_folds=2,
                 loss='rmse',
                 metrics=['mape', 'smape']):
        """Initialize properties."""

        # Forecaster properties
        self.forecaster = forecaster
        self.targets = None
        self.n_samples = 1000

        # Cross-validation properties
        self.test_fraction = test_fraction
        self.n_folds = n_folds
        self.loss = loss
        self.metrics = metrics
        self.predictions_mean = None
        self.predictions_samples = None

        # Training fraction depends on number of folds and test fraction
        print("Training + validation fraction is {}.".format(1 - test_fraction * n_folds))

    def evaluate(self, data, targets=None, verbose=1):
        """Evaluate forecaster."""
        folds = self._generate_folds(data)
        lag = self.forecaster.lag
        horizon = self.forecaster.horizon

        # Forecaster fitting and prediction parameters
        self.targets = targets

        # Set up the metrics dictionary also containing the main loss
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        percentile_names = ['p' + str(x) for x in percentiles]
        metrics = pd.DataFrame(
            columns=[self.loss, ] + self.metrics + percentile_names
        )

        for i, (data_train, data_val) in enumerate(folds):
            # Quietly fit the forecaster to this fold's training set
            forecaster = self.forecaster
            forecaster._is_fitted = False  # Make sure we refit the forecaster
            t0 = time.time()
            forecaster.fit(
                data_train,
                targets=self.targets,
                verbose=0  # Fit in silence
            )

            # Depending on the horizon, we make multiple predictions on the
            # test set and need to create those input output pairs
            j = 0
            inputs = []
            while (j + 1) * horizon + lag <= data_val.shape[1]:
                tmp = []
                for time_series in data_val:
                    tmp.append(time_series[j * horizon:j * horizon + lag, :])
                inputs.append(np.array(tmp))
                j += 1

            # Time series values to be forecasted
            n_horizon = (data_val.shape[1] - lag) // horizon
            if self.targets:
                data_pred = data_val[
                    :, lag:lag + n_horizon * horizon, self.targets
                ]
            else:
                data_pred = data_val[:, lag:lag + n_horizon * horizon, :]

            # Make predictions for each of the input chunks
            predictions_mean, predictions_samples = [], []
            for input_data in inputs:
                prediction = forecaster.predict(
                    input_data,
                    n_samples=self.n_samples
                )
                predictions_mean.append(prediction['mean'])
                predictions_samples.append(prediction['samples'])
            predictions_mean = np.concatenate(predictions_mean, axis=1)
            predictions_samples = np.concatenate(predictions_samples, axis=2)

            # Update the loss for this fold
            metrics_append = {}
            metrics_append[self.loss] = getattr(custom_metrics, self.loss)(
                predictions_mean,
                data_pred
            )

            # Update other performance metrics for this fold
            for metric in self.metrics:
                func = getattr(custom_metrics, metric)
                metrics_append[metric] = func(
                    predictions_mean,
                    data_pred
                )

            # Update coverage metrics for this fold
            for perc in percentiles:
                val_perc = np.percentile(
                    predictions_samples,
                    perc,
                    axis=0
                )
                metrics_append['p' + str(perc)] = custom_metrics.coverage(
                    val_perc,
                    data_pred
                )
            metrics = metrics.append(metrics_append, ignore_index=True)

            # Update the user on the validation status
            duration = round(time.time() - t0)
            if verbose > 0:
                print("Validation fold {} took {} s.".format(i, duration))

        # Clean up the metrics table
        avg = pd.DataFrame(metrics.mean()).T
        avg.index = ['avg.']
        std = pd.DataFrame(metrics.std()).T
        std.index = ['std.']
        metrics = pd.concat([metrics, avg, std])
        metrics = metrics.round(2)
        metrics.index.name = 'fold'

        # Store predictions in case they are needed for plotting
        self.predictions_mean = predictions_mean
        self.predictions_samples = predictions_samples

        return metrics

    def _generate_folds(self, data):
        """Yield a data fold."""
        lag = self.forecaster.lag

        # Find the maximum length of all example time series in the dataset.
        data_length = []
        for time_series in data:
            data_length.append(len(time_series))
        data_length = max(data_length)
        test_length = int(data_length * self.test_fraction)
        train_length = data_length - self.n_folds * test_length

        # Loop over number of folds to generate folds for cross-validation
        # but make sure that the folds do not overlap.
        for i in range(self.n_folds):
            data_train, data_val = [], []
            for time_series in data:
                train_ind = np.arange(
                    -(i + 1) * test_length - train_length,
                    -(i + 1) * test_length
                )
                test_ind = np.arange(
                    -(i + 1) * test_length - lag,
                    -i * test_length
                )
                data_train.append(time_series[train_ind, :])
                data_val.append(time_series[test_ind, :])
            data_train = np.array(data_train)
            data_val = np.array(data_val)

            yield data_train, data_val
