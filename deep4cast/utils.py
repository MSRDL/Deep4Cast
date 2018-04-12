# -*- coding: utf-8 -*-
"""Utilities module for convinience functions."""

import numpy as np


def compute_mape(forecaster, ts, ts_truth):
    """Computes mean absolute percentage error (MAPE) for given model.

    :param forecaster: Neural network forecater
    :type forecater: Forecaster
    :param ts: Time series dataset for training (n_timesteps, n_timeseries)
    :type ts: numpy array
    :param ts_truth: Time series dataset for testing
    :type ts_truth: numpy array

    """

    ts_pred = forecaster.predict(ts)
    return np.mean(np.abs((ts_pred - ts_truth) / ts_truth)) * 100.0
