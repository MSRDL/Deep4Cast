# -*- coding: utf-8 -*-
"""Metrics module for error functions."""

from functools import wraps

import numpy as np


def adjust_for_horizon(metric):
    @wraps(metric)
    def adjusted_metric(data, data_truth):
        # Check if the data has a second dimension (that we interpret as the
        # dimension for multiple steps ahead when doing predictions).
        if len(data.shape) == 4:
            data_truth_adjusted = []
            n = len(data_truth)  # number of time steps in the data
            horizon = data.shape[1]  # extract the length for horizon.
            for i in range(horizon):
                data_truth_adjusted.append(data_truth[i:n-horizon+i+1])
            data_truth_adjusted = np.array(data_truth_adjusted)
            data_truth_adjusted = np.swapaxes(data_truth_adjusted, 0, 1)
        else:
            data_truth_adjusted = data_truth

        return metric(data, data_truth_adjusted)

    return adjusted_metric


def mae(data, data_truth):
    """Computes mean absolute error (MAE)

    :param data: Time series dataset for training (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Prediction corresponding to data
    :type data_truth: numpy array

    """


    return np.mean(np.abs(data - data_truth))


def mape(data, data_truth):
    """Computes mean absolute percentage error (MAPE)

    :param data: Time series dataset for training (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Prediction corresponding to data
    :type data_truth: numpy array

    """

    eps = 1e-16  # Need to make sure that denominator is not zero
    normalization = np.abs(data_truth) + eps

    return np.mean(np.abs(data - data_truth) / normalization) * 100.0


def mse(data, data_truth):
    """Computes mean squared error (MSE)

    :param data: Time series dataset for training (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Prediction corresponding to data
    :type data_truth: numpy array

    """

    return np.mean(np.square((data - data_truth)))


def rmse(data, data_truth):
    """Computes root-mean squared error (RMSE)

    :param data: Time series dataset for training (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Prediction corresponding to data
    :type data_truth: numpy array

    """

    return np.sqrt(mse(data, data_truth))


def smape(data, data_truth):
    """Computes symmetric mean absolute percentage error (SMAPE)

    :param data: Time series dataset for training (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Prediction corresponding to data
    :type data_truth: numpy array

    """

    eps = 1e-16  # Need to make sure that denominator is not zero
    normalization = 0.5 * (np.abs(data) + np.abs(data_truth)) + eps

    return np.mean(np.abs(data - data_truth) / normalization) * 100.0
