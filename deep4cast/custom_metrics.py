# -*- coding: utf-8 -*-
"""Metrics module for error functions."""


import numpy as np

def check_input_shapes(data1, data2):
    """Check if input arrays are of the same shape."""
    if data1.shape != data2.shape:
        raise ValueError("Shape of {} != {}.".format(data1.shape, data2.shape))


def mae(data, data_truth):
    """Computes mean absolute error (MAE)
    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    """
    check_input_shapes(data, data_truth)

    return np.mean(np.abs(data - data_truth))


def mape(data, data_truth):
    """Computes mean absolute percentage error (MAPE)
    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    """
    check_input_shapes(data, data_truth)

    eps = 1e-16  # Need to make sure that denominator is not zero
    normalization = np.abs(data_truth) + eps

    return np.mean(np.abs(data - data_truth) / normalization) * 100.0


def mse(data, data_truth):
    """Computes mean squared error (MSE)
    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    """
    check_input_shapes(data, data_truth)

    return np.mean(np.square((data - data_truth)))


def rmse(data, data_truth):
    """Computes root-mean squared error (RMSE)
    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    """
    check_input_shapes(data, data_truth)

    return np.sqrt(mse(data, data_truth))


def smape(data, data_truth):
    """Computes symmetric mean absolute percentage error (SMAPE)
    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    """
    check_input_shapes(data, data_truth)

    eps = 1e-16  # Need to make sure that denominator is not zero
    normalization = 0.5 * (np.abs(data) + np.abs(data_truth)) + eps

    return np.mean(np.abs(data - data_truth) / normalization) * 100.0


def mase(data, data_truth, insample, freq):
    """Calculates Mean Absolute Scaled Error (MASE)
    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    :param insample: time series in training set (n_timesteps, n_timeseries)
    :type insample: numpy array
    :param freq: frequency or seasonality in the data
    :type freq: integer
    """

    eps = 1e-16  # Need to make sure that denominator is not zero
    normalization = np.mean(np.abs(insample[freq:] - insample[:-freq])) + eps

    return np.mean(np.abs(data - data_truth)) / normalization * 100.0


def msis(data_upper, data_lower, data_truth, insample, freq, alpha=0.05):
    """Computes Mean Scaled Interval Score (MSIS)
    :param data_upper: Predicted upper bound of time series values
    :type data_upper: numpy array
    :param data_lower: Predicted lower bound of time series values
    :type data_lower: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    :param insample: time series in training set (n_timesteps, n_timeseries)
    :type insample: numpy array
    :param freq: frequency or seasonality in the data
    :type freq: integer
    :param alpha: significance level (i.e. 95% confidence interval means alpha = 0.05)
    :type alpha: float
    """

    eps = 1e-16  # Need to make sure that denominator is not zero
    normalization = np.mean(np.abs(insample[freq:] - insample[:-freq])) + eps
    mean_interval_score = np.mean((data_upper - data_lower)
                                  + 2.0 / alpha *
                                  (data_lower - data_truth) *
                                  (data_truth < data_lower)
                                  + 2.0 / alpha *
                                  (data_truth - data_upper) *
                                  (data_truth > data_upper)
                                  )

    return mean_interval_score / normalization * 100.0


def coverage(data_perc, data_truth):
    """Computes coverage rate of the prediction interval.
    :param data_perc: Predicted percentile of time series values
    :type data_perc: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    """

    coverage_percentage = np.mean(data_truth < data_perc)

    return coverage_percentage * 100.0
