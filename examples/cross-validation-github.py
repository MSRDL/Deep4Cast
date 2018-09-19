# -*- coding: utf-8 -*-
"""Executes the whole pipeline for optimization on the Github dataset.
"""
import numpy as np
import pandas as pd

import deep4cast.models as models

from deep4cast.forecasters import Forecaster
from deep4cast.cv import FoldGenerator, VectorScaler, MetricsEvaluator, CrossValidator
from skopt.space import Real, Integer


if __name__ == "__main__":
    # Parameters
    horizon = 90
    lag = 180
    filters = 32
    num_layers = 2
    lr = 0.001
    epochs = 2
    test_fraction = 0.15
    n_folds = 1

    # Load the data from file
    df = pd.read_pickle(
        '../data/processed/github_total_push_events_2011-2018.pkl')
    df = df[:-243]  # Exclude 2018 cause it's not complete

    # Loading the dataset and dropping unncecessary columns
    data = df.drop(['date', 'month', 'day'], axis=1)
    data = data.dropna()

    # We have real-valued and categorial features and we need to make sure that we do the data
    # preparation for them correctly
    real_values = ['count', 'age']
    categorical = list(set(data.columns).difference(set(real_values)))

    # Let's create shifted categorical feature to include information about the future's holiday
    # structure. This will make it easier for our model to do predictions on
    # holidays
    shifted = data[categorical].shift(-horizon)
    shifted = shifted.rename(
        columns={column: column + '_shifted' for column in shifted.columns})
    data = pd.concat([data, shifted], axis=1)

    # Let's also put lagged covariates
    lagged = data[['count']].shift(365 - horizon)
    lagged = lagged.rename(
        columns={column: column + '_annual_lag' for column in lagged.columns}
    )
    data = pd.concat([data, lagged], axis=1)
    data = data.dropna()

    # Format data into numpy array
    data = np.expand_dims(data.values, 0)

    # Forecaster and model setup
    model = models.WaveNet(filters=filters, num_layers=num_layers)
    forecaster = Forecaster(
        model, lag=lag, horizon=horizon, lr=lr, epochs=epochs)

    # build a fold generator
    fold_generator = FoldGenerator(
        data=data,
        targets=[0],
        lag=lag,
        horizon=horizon,
        test_fraction=test_fraction,
        n_folds=n_folds
    )

    # ... and an evaluator
    evaluator = MetricsEvaluator(metrics=['smape', 'pinball_loss', 'coverage'])

    # ... and a data scaler
    scaler = VectorScaler(targets=[0, 1, -1])

    # ... and a cross-validator
    validator = CrossValidator(forecaster,
                               fold_generator,
                               evaluator,
                               scaler,
                               n_ensemble=5)

    # Optimization
    space = [Integer(32, 512, name='filters'),
             Integer(1, 6, name='num_layers'),
             Integer(90, 730, name='lag'),
             Integer(10, 500, name='epochs'),
             Integer(4, 128, name='batch_size'),
             Real(10**-5, 10**-3, "log-uniform", name='lr')]

    opt_results = validator.optimize(space, 'pinball_loss', n_calls=50)
