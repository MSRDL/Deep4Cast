"""Cross-validation and optimization module.
This module provides access to tools for cross-validation and optimization of
forecasters using Bayesian Optimization.
"""
import time

import numpy as np
import pandas as pd

from . import utils
from . import custom_metrics
from inspect import getfullargspec
from skopt.utils import use_named_args
from skopt import gp_minimize

__FOLD_GENERATOR_ARGS__ = ['lag']
__MODEL_ARGS__ = ['filters', 'units', 'num_layers']
__FORECASTER_ARGS__ = ['epochs', 'batch_size']
__OPTIMIZER_ARGS__ = ['lr']


class CrossValidator():
    """Temporal cross-validator class.

    This class performs temporal (causal) cross-validation similar to the
    approach in https://robjhyndman.com/papers/cv-wp.pdf.

    :param forecaster: Forecaster.
    :type forecaster: A forecaster class
    :param fold_generator: Fold generator.
    :type fold_generator: A fold generator class
    :param evaluator: Evaluator.
    :type evaluator: An evaluator class
    :param scaler: Scaler.
    :type scaler: A scaler class
    :param optimizer: Optimizer.
    :type optimizer: An optimizer class
    """

    def __init__(self,
                 forecaster,
                 fold_generator,
                 evaluator,
                 scaler=None,
                 optimizer=None):
        """Initialize properties."""
        self.forecaster = forecaster
        self.fold_generator = fold_generator  # Must be a generator
        self.evaluator = evaluator
        self.scaler = scaler

        # Optimizer arguments
        self.space = None

    def evaluate(self, n_samples=1000, verbose=True):
        """Evaluate forecaster."""
        self.evaluator.reset()  # Make sure we have a clean evaluator

        for X_train, X_test, y_train, y_test in self.fold_generator():
            # Set up the forecaster
            forecaster = self.forecaster
            t0 = time.time()

            # Transform the data
            if self.scaler:
                X_train = self.scaler.fit_transform_x(X_train)
                X_test = self.scaler.transform_x(X_test)
                y_train = self.scaler.fit_transform_y(y_train)

            # Quietly fit the forecaster to this fold's training set
            forecaster.fit(X_train, y_train, verbose=0)

            # Generate predictions but only on necessary data points
            y_pred_samples = forecaster.predict(
                X_test[::self.fold_generator.horizon],
                n_samples=n_samples
            )

            # Transform the samples back
            y_pred_samples = self.scaler.inverse_transform_y(y_pred_samples)

            # Reformat samples for evaluation into time series format
            eval_samples = []
            for i in range(y_pred_samples.shape[1]):
                eval_samples.append(y_pred_samples[:, i, :, :])
            eval_samples = np.concatenate(eval_samples, axis=1)

            # Reformat the data into time series forcast for evaluation
            y_eval = y_test[::self.fold_generator.horizon]
            y_eval = np.reshape(y_eval, y_eval.shape[0] * y_eval.shape[1])
            y_eval = np.atleast_2d(y_eval).T

            # Evaluate forecaster performance
            self.evaluator.evaluate(eval_samples, y_eval, verbose=verbose)
            if verbose:
                print('Evaluation took {} seconds.'.format(time.time() - t0))

        return self.evaluator.tearsheet

    def optimize(self, space, metric, n_calls=10, n_samples=1000):
        """Optimize the forecaster parameters."""
        args = self.get_args()

        @use_named_args(space)
        def objective(**params):
            """This is the function that we build fgor the optimizer to
            optimizer."""
            for key, value in params.items():
                if key in args['fold_generator'] and key in __FOLD_GENERATOR_ARGS__:
                    setattr(self.fold_generator, key, value)
                elif key in args['model'] and key in __MODEL_ARGS__:
                    setattr(self.forecaster.model, key, value)
                elif key in args['optimizer'] and key in __OPTIMIZER_ARGS__:
                    setattr(self.forecaster._optimizer, key, value)
                elif key in args['forecaster'] and key in __FORECASTER_ARGS__:
                    setattr(self.forecaster, key, value)
                else:
                    raise ValueError('{} not a valid argument'.format(key))

            # Make sure the forecaster is refitted and reset
            self.forecaster.reset()

            # Tearsheet is the summary of this CV run
            print(params)
            tearsheet = self.evaluate(n_samples=n_samples, verbose=False)
            print(np.mean(tearsheet[metric]))

            # We take the mean value of the tearsheet metric that we care
            # about as optimization objective
            return np.mean(tearsheet[metric])

        # Optimize everything
        res_gp = gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            random_state=0
        )

        return res_gp

    def get_args(self):
        """Return the parameters that the forecaster can take."""
        fold_generator_args = getfullargspec(self.fold_generator.__class__).args
        model_args = getfullargspec(self.forecaster.model.__class__).args
        forecaster_args = getfullargspec(self.forecaster.__class__).args
        optimizer_args = getfullargspec(self.forecaster._optimizer.__class__).args
        return {
            'fold_generator': fold_generator_args,
            'model': model_args,
            'forecaster': forecaster_args,
            'optimizer': optimizer_args
        }


class FoldGenerator():
    """Cross-validation fold generator class.

    """

    def __init__(self, data, targets, lag, horizon, test_fraction, n_folds):
        """Initialize properties."""
        self.data = data
        self.targets = targets
        self.lag = lag
        self.horizon = horizon
        self.test_fraction = test_fraction
        self.n_folds = n_folds

    def __call__(self):
        return self.generate_folds()

    def generate_folds(self):
        """Yields a data fold."""
        # Find the maximum length of all example time series in the dataset.
        data_length = []
        for time_series in self.data:
            data_length.append(len(time_series))
        data_length = max(data_length)
        test_length = int(data_length * self.test_fraction)
        train_length = data_length - self.n_folds * test_length

        # Loop over number of folds to generate folds for cross-validation
        # but make sure that the folds do not overlap.
        for i in range(self.n_folds):
            data_train, data_test = [], []
            for time_series in self.data:
                train_ind = np.arange(
                    -(i + 1) * test_length - train_length,
                    -(i + 1) * test_length
                )
                test_ind = np.arange(
                    -(i + 1) * test_length - self.lag,
                    -i * test_length
                )
                data_train.append(time_series[train_ind, :])
                data_test.append(time_series[test_ind, :])
            data_train = np.array(data_train)
            data_test = np.array(data_test)

            # Sequentialize dataset
            X_train, y_train = utils.sequentialize(
                data_train,
                self.lag,
                self.horizon,
                targets=self.targets
            )
            X_test, y_test = utils.sequentialize(
                data_test,
                self.lag,
                self.horizon,
                targets=self.targets
            )
            yield X_train, X_test, y_train, y_test


class MetricsEvaluator():
    """Metrics evaluator  class.

    Evaluates a list of metrics on a dataset.

    """

    def __init__(self, metrics, filename=None):
        """Initialize properties."""
        self.metrics = metrics
        self.tearsheet = pd.DataFrame(columns=self.metrics)
        self.filename = filename

    def evaluate(self, y_samples, y_truth, verbose=False):
        """Calculate all the metrics and store them in tearsheet."""
        eval_results = {}
        for metric in self.metrics:
            try:
                eval_func = getattr(custom_metrics, metric)
                eval_results[metric] = eval_func(y_samples, y_truth)
                if verbose:
                    print('Results for {} is {}.'.format(
                        metric,
                        eval_results[metric])
                    )
            except:
                print('{} not a valid metric'.format(metric))
        self.tearsheet = self.tearsheet.append(eval_results, ignore_index=True)
        if self.filename:
            self.to_pickle()

    def to_pickle(self):
        """Store results."""
        self.tearsheet.to_pickle(self.filename)

    def reset(self):
        """Reset internal state."""
        self.tearsheet = pd.DataFrame(columns=self.metrics)
