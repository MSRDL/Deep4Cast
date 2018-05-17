# -*- coding: utf-8 -*-
"""Validation module for evaluating forecasters.

This module provides access to validation classes that can be used to
evaluate forecasting performance.

"""
import numpy as np

from . import metrics
from .forecasters import Forecaster


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
                 data,
                 train_frac=0.5,
                 n_folds=5,
                 loss='mse'):
        """Initialize properties."""
        self.forecaster = forecaster
        self.data = data
        self.train_frac = train_frac
        self.n_folds = n_folds
        self.loss = loss

    def __call__(self, params, verbose=False):
        """Support functional API for this class to be able to interface
        easility with hyperparameter optimizers.

        """

        return self.evaluate(params, verbose=verbose)

    def evaluate(self, params, verbose=True):
        """Evaluate forecaster with forecaster parameters params.

        :param params: Dictionary that contains parameters for forecaster.
        :type params: dict

        """

        # Instantiate the appropriate loss metric and get the folds for
        # evaluating the forecaster. We want to use a generator here to save
        # some space.
        folds = self._generate_folds(params['lag'])

        train_losses, test_losses = [], []
        for i, (data_train, data_test) in enumerate(folds):
            # Quietly fit the forecaster
            forecaster = self.forecaster(**params)
            forecaster.fit(data_train, verbose=0)

            # Calculate forecaster performance
            train_predictions = forecaster.predict(data_train)['mean']
            test_predictions = forecaster.predict(data_test)['mean']

            train_actuals = data_train[params['lag']:]
            test_actuals = data_test[params['lag']:]

            # Make sure the loss function knows about the multi-step
            # forecasting procedure.
            loss = getattr(metrics, self.loss)
            if forecaster.horizon > 1:
                loss = metrics.adjust_for_horizon(loss)

            train_loss = round(loss(train_predictions, train_actuals), 2)
            test_loss = round(loss(test_predictions, test_actuals), 2)

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            # Report progress if requested
            if verbose:
                print(
                    'Fold {}: Train {}:{}, Test {}:{}'.format(
                        i,
                        self.loss,
                        train_loss,
                        self.loss,
                        test_loss
                    )
                )

        # We only need some loss statistics. We use the name 'loss' in this
        # dictionary to denote the main quantity of interest, because
        # hyperopt expect a dictionary with a 'loss' key.
        scores = {
            'loss': np.mean(test_losses),
            'loss_std': np.std(test_losses),
            'loss_min': np.min(test_losses),
            'loss_max': np.max(test_losses),
            'train_loss': np.mean(train_losses),
            'train_loss_std': np.std(train_losses),
            'train_loss_min': np.min(train_losses),
            'train_loss_max': np.max(train_losses)
        }

        return scores

    def _generate_folds(self, lag):
        """Yield a the data folds."""
        train_length = int(len(self.data) * self.train_frac)
        fold_length = (len(self.data) - train_length) // self.n_folds

        # Loopp over number of folds to generate folds for cross-validation
        # but make sure that the train and test part of the time series
        # dataset overlap appropriately to account for the lag window.
        for i in range(self.n_folds):
            a = i * fold_length
            b = (i + 1) * fold_length
            lb = lag
            data_train = self.data[a:train_length + a]
            data_test = self.data[train_length + a - lb:train_length + b]

            yield (data_train, data_test)
