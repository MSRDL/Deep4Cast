# -*- coding: utf-8 -*-
"""Time series forecasting module.
This module provides access to forecasters that can be fit multivariate time
series given as numpy arrays.
"""
from inspect import getargspec

import numpy as np
import keras.optimizers

from . import custom_losses


class Forecaster():
    def __init__(self,
                 model,
                 lag,
                 horizon,
                 loss='heteroscedastic_gaussian',
                 optimizer='adam',
                 batch_size=16,
                 epochs=100,
                 **kwargs):
        """Initialize properties."""
        # Neural network model attributes
        self.model = model  # Neural network architecture (Keras model)
        self.lag = lag   # Lookback window (length of input)
        self.horizon = horizon  # Forecasting horizon (length of output)

        # Optimizer attributes
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.history = None

        # Configure optimizer
        allowed_kwargs = self.get_optimizer_args()
        for key, value in kwargs.items():
            if key not in allowed_kwargs:
                raise ValueError('{} not a valid argument.'.fomat(key))
        self.set_optimizer_args(kwargs)

        # Boolean checks
        self.is_fitted = False

    def fit(self, X, y, verbose=0):
        """Fit model to data."""
        # Make sure the data type is float32 for optimal performance of all
        # Keras backends.
        X = X.astype('float32')
        y = y.astype('float32')

        # Prepare model output shape based on loss function type to handle
        # loss functions that requires multiple parameters such as the
        # heteroscedsatic Gaussian
        if isinstance(self.loss, str):
            self._loss = getattr(custom_losses, self.loss)(
                n_dim=y.shape[2]
            )
        else:
            self._loss = self.loss
        loss_dim_factor = self._loss.dim_factor
        output_shape = (y.shape[1], y.shape[2] * loss_dim_factor)

        # Need to handle the case where the model is fitted for more epochs
        # after it has already been fitted
        if not self.is_fitted:
            # Set up the model based on internal model class
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
            verbose=verbose,
        )

        # Change state to fitted so that other methods work correctly
        self.is_fitted = True

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
        X = np.repeat(X, [n_samples]*len(X), axis=0)

        # Make predictions for parameters of pdfs then sample from pdfs
        predictions = self.model.predict(X, self.batch_size)
        predictions = self._loss.sample(predictions, n_samples=1)

        # Reorganize prediction samples into 3D array
        reshuffled_predictions = []
        for i in range(n_samples):
            block = predictions[i * block_size:(i + 1) * block_size]
            reshuffled_predictions.append(block)
        predictions = np.array(reshuffled_predictions)

        return predictions

    def check_is_fitted(self):
        """Raise error if model has not been fitted to data."""
        if not self.is_fitted:
            raise ValueError('The model has not been fitted.')

    @property
    def optimizer(self):
        """Return the optimizer name."""
        return self._optimizer.__class__.__name__

    @optimizer.setter
    def optimizer(self, optimizer):
        """Instantiate the optimizer."""
        optimizer_class = getattr(keras.optimizers, optimizer)
        self._optimizer = optimizer_class()

    def get_optimizer_args(self):
        """Get optimizer parameters."""
        args = getargspec(self._optimizer.__class__)[0]
        args.remove('self')
        return args

    def set_optimizer_args(self, params):
        """Set optimizer parameters."""
        optimizer_class = self._optimizer.__class__
        optimizer_args = getargspec(optimizer_class)[0]
        for key, value in params.items():
            if key in optimizer_args:
                setattr(self._optimizer, key, value)
