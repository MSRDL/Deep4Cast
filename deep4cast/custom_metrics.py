# -*- coding: utf-8 -*-
"""Metrics module for error functions."""

import numpy as np


def corr(data_samples, data_truth):
    """Computes the empirical correlation betnween actuals and predictions
    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    """
    data = np.mean(data_samples, axis=0)

    return round(np.corrcoeff(data, data_truth, rowvar=False), 2)


def mae(data_samples, data_truth):
    """Computes mean absolute error (MAE)
    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    """
    data = np.mean(data_samples, axis=0)

    return round(np.mean(np.abs(data - data_truth)), 2)


def mape(data_samples, data_truth):
    """Computes mean absolute percentage error (MAPE)
    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    """
    data = np.mean(data_samples, axis=0)

    eps = 1e-16  # Need to make sure that denominator is not zero
    norm = np.abs(data_truth) + eps

    return round(np.mean(np.abs(data - data_truth) / norm) * 100.0, 2)


def smape(data_samples, data_truth):
    """Computes symmetric mean absolute percentage error (SMAPE)
    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    """
    data = np.mean(data_samples, axis=0)

    eps = 1e-16  # Need to make sure that denominator is not zero
    norm = 0.5 * (np.abs(data) + np.abs(data_truth)) + eps

    return round(np.mean(np.abs(data - data_truth) / norm) * 100.0, 2)


def mse(data_samples, data_truth):
    """Computes mean squared error (MSE)
    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    """
    data = np.mean(data_samples, axis=0)

    return round(np.mean(np.square((data - data_truth))), 2)


def rmse(data_samples, data_truth):
    """Computes root-mean squared error (RMSE)
    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    """
    data = np.mean(data_samples, axis=0)

    return round(np.sqrt(mse(data, data_truth)), 2)


def rse(data_samples, data_truth):
    """Computes root relative squared error (RSE)
    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    """
    data = np.mean(data_samples, axis=0)

    norm = np.sqrt(np.sum(data_truth - np.mean(data_truth, axis=0)))

    return np.sqrt(np.sum(np.square(data - data_truth))) / norm


def coverage(data_samples, data_truth, percentiles=None):
    """Computes coverage rates of the prediction interval.
    :param data_samples: Samples of time series values
    :type data_samples: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    :param percentiles: Percentiles to compute coverage for
    :type percentiles: list
    """
    if percentiles is None:
        percentiles = [1, 5, 25, 50, 75, 95, 99]

    data_perc = np.percentile(data_samples, q=percentiles, axis=0)
    coverage_percentages = []
    for perc in data_perc:
        coverage_percentages.append(np.mean(data_truth <= perc) * 100.0)

    return coverage_percentages, percentiles


def pinball_loss(data_samples, data_truth, percentiles=None):
    """Computes pinball loss.
    :param data_samples: Samples of time series values
    :type data_samples: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    :param percentiles: Percentiles to compute coverage for
    :type percentiles: list
    """
    if percentiles is None:
        percentiles = np.linspace(0, 100, 101)

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
