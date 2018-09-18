# -*- coding: utf-8 -*-
"""Utilities module.

"""
import numpy as np


def check_data_format(data, horizon):
    """Raise error if data has incorrect format."""
    # Check if data has any NaNs.
    if np.isnan([np.isnan(x).any() for x in data]).any():
        raise ValueError('data should not contain NaNs.')

    # Check if data is long enough for forecasting horizon.
    if np.array([len(x) <= horizon for x in data]).any():
        raise ValueError('Time series must be longer than horizon.')


def sequentialize(data, lag, horizon, targets=None):
    """Sequentialize time series array.
    Create two numpy arrays, one for the windowed input time series X
    and one for the corresponding output values that need to be
    predicted.
    """

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
            if targets:
                y.append(forecast_ts[:, targets])
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

    # Remove NaNs that occur during windowing
    X = X[~np.isnan(y)[:, 0, 0]]
    y = y[~np.isnan(y)[:, 0, 0]]

    return X, y
