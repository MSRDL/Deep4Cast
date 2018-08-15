# -*- coding: utf-8 -*-
"""Metrics module for error functions."""


import numpy as np


def check_input_shapes(data1, data2):
    """Check if input arrays are of the same shape."""
    if data1.shape != data2.shape:
        raise ValueError(
            "Shape of {} != {}.".format(data1.shape, data2.shape)
            )


def corr(data, data_truth):
    """Computes the empirical correlation betnween actuals and predictions
    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    """
    check_input_shapes(data, data_truth)

    return np.corrcoeff(data, data_truth, rowvar=False)


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


def rse(data, data_truth):
    """Computes root relative squared error (RSE)
    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    """
    check_input_shapes(data, data_truth)

    normalization = np.sqrt(np.sum(data_truth - np.mean(data_truth, axis=0)))

    return np.sqrt(np.sum(np.square(data - data_truth))) / normalization


def coverage(data_samples, data_truth, percentiles):
    """Computes coverage rates of the prediction interval.
    :param data_samples: Samples of time series values
    :type data_samples: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    :param percentiles: Percentiles to compute coverage for
    :type percentiles: list
    """

    data_perc = np.percentile(data_samples, q=percentiles, axis=0)
    coverage_percentages = []
    for perc in data_perc:
        coverage_percentages.append(np.mean(data_truth <= perc) * 100.0)

    return coverage_percentages


def pinball_loss(data_samples, data_truth, percentiles):
    """Computes pinball loss.
    :param data_samples: Samples of time series values
    :type data_samples: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    :param percentiles: Percentiles to compute coverage for
    :type percentiles: list
    """
    num_steps = data_samples.shape[1]

    # Calculate percentiles
    data_perc = np.percentile(data_samples, q=percentiles, axis=0)

    # Calculate mean pinball loss
    total = 0
    for perc, q in zip(data_perc, percentiles):
        # Calculate upper and lower branch of pinball loss
        upper = data_truth - perc
        lower = perc - data_truth
        upper = np.sum(q/100.0*upper[upper >= 0])
        lower = np.sum((1-q/100.0)*lower[lower > 0])
        total += (upper + lower)/num_steps

    # Add overall mean pinball loss
    return round(total / len(percentiles), 2)


def std_smape(data_samples, data_truth):
    """Computes mean absolute percentage error (MAPE) of 
    sample standard deviation.
    :param data_samples: Samples of time series values
    :type data_samples: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    """  
    eps = 1e-16

    mean = np.mean(data_samples, axis=0)
    std = np.std(data_samples, axis=0)

    abs_diff = np.abs(data_truth - mean)
    normalization = 0.5*(np.abs(abs_diff) + np.abs(std)) + eps

    return np.mean(np.abs(std - abs_diff) / normalization) * 100.0
