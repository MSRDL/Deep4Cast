# -*- coding: utf-8 -*-
"""Metrics module for error functions."""

import numpy as np


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
