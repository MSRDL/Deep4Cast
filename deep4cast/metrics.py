# -*- coding: utf-8 -*-
"""Metrics module for error functions."""

from functools import wraps

import numpy as np


def adjust_for_horizon(metric):
    @wraps(metric)
    def adjusted_metric(data, data_truth, *param):
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

        return metric(data, data_truth_adjusted, *param)

    return adjusted_metric


def mae(data, data_truth):
    """Computes mean absolute error (MAE)

    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array

    """


    return np.mean(np.abs(data - data_truth))


def mape(data, data_truth):
    """Computes mean absolute percentage error (MAPE)

    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array

    """

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

    return np.mean(np.square((data - data_truth)))


def rmse(data, data_truth):
    """Computes root-mean squared error (RMSE)

    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array

    """

    return np.sqrt(mse(data, data_truth))


def smape(data, data_truth):
    """Computes symmetric mean absolute percentage error (SMAPE)

    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array

    """

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
    :param freq: frequency or seasonality in the data (i.e. 12 for monthly series)
    :type freq: integer

    """

    eps = 1e-16  # Need to make sure that denominator is not zero
    normalization = np.mean(np.abs(insample[freq:] - insample[:-freq])) + eps

    return np.mean(np.abs(data - data_truth)) / normalization * 100.0


def msis(data_upper, data_lower, data_truth, insample, freq, alpha = 0.05):
    """Computes Mean Scaled Interval Score (MSIS)

    :param data_upper: Predicted upper bound of time series values 
    :type data_upper: numpy array
    :param data_lower: Predicted lower bound of time series values
    :type data_lower: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    :param insample: time series in training set (n_timesteps, n_timeseries)
    :type insample: numpy array
    :param freq: frequency or seasonality in the data (i.e. 12 for monthly series)
    :type freq: integer
    :param alpha: significance level (i.e. 95% confidence interval means alpha = 0.05) 
    :type alpha: float

    """

    eps = 1e-16  # Need to make sure that denominator is not zero
    normalization = np.mean(np.abs(insample[freq:] - insample[:-freq])) + eps
    mean_interval_score = np.mean((data_upper - data_lower) \
                                   + 2.0/alpha*(data_lower - data_truth)*(data_truth < data_lower) \
                                   + 2.0/alpha*(data_truth - data_upper)*(data_truth > data_upper))
    
    return mean_interval_score / normalization * 100.0


def coverage(data_upper, data_lower, data_truth):
    """Computes coverage rate of the prediction interval.

    :param data_upper: Predicted upper bound of time series values 
    :type data_upper: numpy array
    :param data_lower: Predicted lower bound of time series values
    :type data_lower: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array

    """

    coverage_percentage = np.mean(1.0*(data_truth > data_lower)*(data_truth < data_upper))
    
    return coverage_percentage * 100.0

