# -*- coding: utf-8 -*-
"""Time series forecasting module.
This module provides access to forecasters that can be fit multivariate time
series given as numpy arrays.
"""
from inspect import getargspec

import time
import numpy as np
import keras.optimizers
from keras.callbacks import EarlyStopping

from . import custom_losses, metrics
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
    :param dropout_rate: Probability of dropping a node during Dropout.
    :type dropout_rate: float
    """

    def __init__(self,
                 topology,
                 lag: int,
                 horizon: int,
                 loss='mse',
                 optimizer='sgd',
                 batch_size=16,
                 max_epochs=1000,
                 dropout_rate=None,
                 **kwargs):
        """Initialize properties."""

        # Attributes related to neural network model
        self.model_class = SharedLayerModel
        self.topology = self._build_topology(topology)
        self.dropout_rate = dropout_rate
        self._model = None

        # Attributes related to model training
        self.optimizer = optimizer
        self.set_optimizer_args(kwargs)

        self.lag = lag
        self.horizon = horizon
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.loss = loss
        self.history = None
        self._loss = None
        self.targets = None

        # Attributes related to input data (these are set during fitting)
        self._features_means = None
        self._features_scales = None
        self._targets_means = None
        self._targets_scales = None

        # Boolean checks
        self._is_fitted = False
        self._is_normalized = False

    def fit(self,
            data,
            targets=None,
            normalize=False,
            val_frac=0.1,
            patience=5,
            verbose=0):
        """Fit model to data.
        :param data: Time series array of shape (n_steps, n_variables).
        :type data: numpy.array
        :param targets: Dict that contains the indices of targets and
        covariates.
        :type targets: dict
        :param val_frac: Fraction of data points to set aside for validation.
        :type val_frac: float
        :param patience: number of epochs to wait before early stopping,
        :type patience: float
        :param verbose: Verbosity mode. 0 = silent, 1 = progress bar,
            2 = one line per epoch.
        :type verbose: int
        """
        # Store a dictionary of lists that contain the indicies of
        # target variables, dynamic, and static covariates.
        self.targets = targets

        # Check if data doesn't contains NaNs and such
        self._check_data_format(data)

        # Make sure the floats have the correct format
        data_train = self._convert_to_float32(data)

        # The model expects input-output data pairs, so we create them from
        # the time series arrary by windowing. Xs are 3D tensors
        # of shape number of batch_size, timesteps, input_dim and
        # ys are 2D tensors of lag * dimensionality.
        X, y = self._sequentialize(data_train)

        # Remove NaN's from windowing
        X = X[~np.isnan(y)[:, 0, 0]]
        y = y[~np.isnan(y)[:, 0, 0]]

        # Standardize the data before feeding it into the model
        X, y = self._normalize(X, y)

        # Shuffle training sequences for validation set creation
        inds = np.arange(len(X))
        np.random.shuffle(inds)
        X = X[inds]
        y = y[inds]

        # Split into training and validation for early stopping
        n_val = int(len(X) * val_frac)
        X_train = X[:-n_val]
        y_train = y[:-n_val]
        X_val = X[-n_val:]
        y_val = y[-n_val:]

        # Prepare model output shape based on loss function
        # This is to handle loss functions that requires multiple parameters
        # such as the heteroscedsatic Gaussian
        self._loss = getattr(custom_losses, self.loss)(n_dim=y.shape[2])
        loss_dim_factor = self._loss.dim_factor
        output_shape = (y.shape[1], y.shape[2] * loss_dim_factor)

        # Set up the model based on internal model class
        self._model = self.model_class(
            input_shape=X.shape[1:],
            output_shape=output_shape,
            topology=self.topology,
            dropout_rate=self.dropout_rate
        )

        # Keras needs to compile the computational graph before fitting
        self._model.compile(
            loss=self._loss,
            optimizer=self._optimizer
        )

        # Set up early stopping callback
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience)

        # Print the model topology and parameters before fitting
        self.history = self._model.fit(
            X_train,
            y_train,
            shuffle=False,
            batch_size=int(self.batch_size),
            epochs=int(self.max_epochs),
            validation_data=(X_val, y_val),
            callbacks=[es],
            verbose=verbose
        )

        # Change state to fitted so that other methods work correctly
        self._is_fitted = True

        # Store the model summary
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
        # Check if model is fitted and data doesn't contains NaNs and such
        self._check_is_fitted()
        self._check_data_format(data)

        # Now only use the last windows from the input sequences
        data_pred = []
        for time_series in data:
            data_pred.append(time_series[-self.lag:, :])
        data_pred = np.array(data_pred)

        # Make sure the floats have the correct format
        data_pred = self._convert_to_float32(data_pred)

        means, stds, lower_quantiles, upper_quantiles = [], [], [], []
        samples = []
        for time_series in data_pred:
            time_series = np.expand_dims(time_series, 0)
            # The model expects input-output data pairs, so we create them from
            # the time series arrary by windowing. Xs are 3D tensors
            # of shape number of batch_size, timesteps, input_dim and
            # ys are 2D tensors of lag * dimensionality.
            X, y = self._sequentialize(time_series)

            # Standardize the data before feeding it into the model
            X, y = self._normalize(X, y, locked=True)

            # If uncertainty is False, only do one sample prediction
            if not self.dropout_rate:
                n_samples = 1

            # Repeat the prediction n_samples times to generate samples from
            # approximate posterior predictive distribution.
            block_size = len(X)
            X = np.repeat(X, [n_samples for _ in range(len(X))], axis=0)

            # Make predictions for parameters of pdfs then sample from pdfs
            raw_predictions = self._model.predict(X, self.batch_size)
            raw_predictions = self._loss.sample(raw_predictions)

            # Take care of means and standard deviations
            predictions = self._unnormalize_targets(raw_predictions)

            # Calculate staticts on predictions
            reshuffled_predictions = []
            for i in range(n_samples):
                block = predictions[i * block_size:(i + 1) * block_size]
                reshuffled_predictions.append(block)
            predictions = np.array(reshuffled_predictions)

            # predictions = np.array(np.vsplit(predictions, n_samples))
            mean = np.mean(predictions, axis=0)
            std = np.std(predictions, axis=0)
            lower_quantile = np.percentile(
                predictions, quantiles[0], axis=0
            )
            upper_quantile = np.percentile(
                predictions, quantiles[1], axis=0
            )

            means.append(mean)
            stds.append(std)
            lower_quantiles.append(lower_quantile)
            upper_quantiles.append(upper_quantile)
            samples.append(predictions)

        means = np.array(means)[:, 0, :, :]
        stds = np.array(stds)[:, 0, :, :]
        lower_quantiles = np.array(lower_quantiles)[:, 0, :, :]
        upper_quantiles = np.array(upper_quantiles)[:, 0, :, :]
        samples = np.array(samples)[:, :, 0, :]
        samples = np.swapaxes(samples, 0, 1)

        return {'mean': means,
                'std': stds,
                'lower_quantile': lower_quantiles,
                'upper_quantile': upper_quantiles,
                'samples': samples}

    @staticmethod
    def _build_topology(topology):
        """Return topology depending on user input (str or list)."""
        if isinstance(topology, str):
            return get_topology(topology)
        if isinstance(topology, list):
            return topology

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
        # Redefine variable to keep further visual noise low
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

    def _normalize(self, X, y, locked=False):
        """normalize numpy array.
        :param data: Input time series.
        :type data: numpy array
        :param data: Boolean that locks down the scales of the data.
        :type data: boolean
        """

        if not locked:
            self._features_means = np.mean(X, axis=0)
            self._features_scales = np.std(X, axis=0)
            self._target_means = np.mean(y, axis=0)
            self._target_scales = np.std(y, axis=0)
            self._is_normalized = True

        # Standardize the data
        X_norm = np.copy(X)
        y_norm = np.copy(y)
        X_norm = (X_norm - self._features_means) / self._features_scales
        y_norm = (y_norm - self._target_means) / self._target_scales

        return X_norm, y_norm

    def _unnormalize_targets(self, y):
        """Un-normalize numpy array."""
        # Unnormalize, taking presence of covariates into account.
        return y * self._target_scales + self._target_means

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


class TemporalCrossValidator():
    """Temporal cross-validator class.

    This class performs temporal (causal) cross-validation similar to the
    approach in  https://robjhyndman.com/papers/cv-wp.pdf.

    :param forecaster: Forecaster.
    :type forecaster: A forecaster class
    :param data: Data set to used for cross-validation.
    :type data: numpy array
    :param train_frac: Fraction of data to be used for training per fold.
    :type train_frac: float
    :param n_folds: Number of temporal folds.
    :type n_folds: int
    :param loss: The kind of loss used for evaluating the forecaster on folds.
    :type loss: string

    """

    def __init__(self,
                 forecaster: Forecaster,
                 n_folds=5,
                 loss='mse'):
        """Initialize properties."""
        self.forecaster = forecaster
        self.n_folds = n_folds
        self.loss = loss

    def evaluate(self, data, targets=None, patience=10, verbose=True):
        """Evaluate forecaster with forecaster parameters params.

        :param params: Dictionary that contains parameters for forecaster.
        :type params: dict

        """

        # Instantiate the appropriate loss metric and get the folds for
        # evaluating the forecaster. We want to use a generator here to save
        # some space.
        folds = self._generate_folds(data)

        losses = []
        for i, (data_train, data_test) in enumerate(folds):
            # Quietly fit the forecaster
            forecaster = self.forecaster
            t0 = time.time()
            forecaster.fit(
                data_train,
                targets=targets,
                patience=patience,
                verbose=0
            )
            duration = time.time() - t0

            # Calculate forecaster performance
            predictions = forecaster.predict(data_train)
            loss = getattr(metrics, self.loss)
            loss = round(metrics.mse(
                predictions['mean'], data_test[:, :, targets]),
                2
            )
            losses.append(loss)

            # Report progress if requested
            if verbose:
                print(
                    'Fold {}: Test {}:{}'.format(
                        i,
                        self.loss,
                        loss
                    )
                )

        # We only need some loss statistics. We use the name 'loss' in this
        # dictionary to denote the main quantity of interest, because
        # hyperopt expect a dictionary with a 'loss' key.
        scores = {
            'loss': np.mean(losses),
            'loss_std': np.std(losses),
            'loss_min': np.min(losses),
            'loss_max': np.max(losses),
            'training_time': duration
        }

        return scores

    def _generate_folds(self, data):
        """Yield a data fold."""
        horizon = self.forecaster.horizon
        train_length = data.shape[1] - horizon * self.n_folds

        # Loop over number of folds to generate folds for cross-validation
        # but make sure that the train and test part of the time series
        # dataset overlap appropriately to account for the lag window.
        for i in range(self.n_folds):
            data_train, data_test = [], []
            for time_series in data:
                data_train.append(
                    time_series[i * horizon: train_length + i * horizon, :]
                )
                data_test.append(
                    time_series[train_length + i *
                                horizon: train_length + (i + 1) * horizon, :]
                )
            data_train = np.array(data_train)
            data_test = np.array(data_test)

            yield (data_train, data_test)
