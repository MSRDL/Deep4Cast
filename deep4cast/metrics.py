# -*- coding: utf-8 -*-
"""Metrics module for error functions."""

from functools import wraps

import numpy as np


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
    :param freq: frequency or seasonality in the data (i.e. 12 for monthly series)
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


def coverage(data_upper, data_lower, data_truth):
    """Computes coverage rate of the prediction interval.

    :param data_upper: Predicted upper bound of time series values 
    :type data_upper: numpy array
    :param data_lower: Predicted lower bound of time series values
    :type data_lower: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array

    """

    coverage_percentage = np.mean(
        1.0 * (data_truth > data_lower) * (data_truth < data_upper))

    return coverage_percentage * 100.0


def print_model_performance_mean_accuracy(data, data_truth,
                                          metric_list=['mape', 'smape'],
                                          freq=12, ts_train=None):
    """Print out model performance on prediction accuracy

    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    :param metric_list: names of metrics to measure accuracy, e.g. mape, smape, mase
    :type metric_list: string or list, e.g. 'mape' or ['mape','smape','mase']
    :param freq: frequency or seasonality in the data (i.e. 12 for monthly series)
    :type freq: integer
    :param ts_train: time series in training set (n_timesteps, n_timeseries)
    :type ts_train: numpy array

    """

    # check if metric_list is a list; if not, convert to a list.
    if (not isinstance(metric_list, list)):
        metric_list = [metric_list]

    metric_value = []
    for i in metric_list:
        if (i == 'mape'):
            name = 'Mean Absolute Percentage Error'
            val = mape(data, data_truth)
            print('\t {0}: {1:.1f}%'.format(name, val))
            metric_value.append(val)
        elif(i == 'smape'):
            name = 'Symmetric Mean Absolute Percentage Error'
            val = smape(data, data_truth)
            print('\t {0}: {1:.1f}%'.format(name, val))
            metric_value.append(val)
        elif (i == 'mase'):
            name = 'Mean Absolute Scaled Error'
            val = mase(data, data_truth, ts_train, freq)
            print('\t {0}: {1:.1f}%'.format(name, val))
            metric_value.append(val)

    return metric_value


def print_model_performance_uncertainty(pred_samples, data_truth,
                                        metric_list='coverage', freq=12,
                                        confidence_level=0.95,
                                        ts_train=None, verbose=True):
    """Print out model performance on uncertainty

    :param pred_samples: Prediction samples of time series (n_timesteps, n_timeseries)
    :type pred_samples: numpy array
    :param data_truth: Ground truth time series values (n_timesteps, n_timeseries)
    :type data_truth: numpy array
    :param metric_list: names of metrics to measure uncertainty, e.g. 'msis', 'coverage'
    :type metric_list: string or list, e.g. 'coverage' or ['msis','coverage']
    :param freq: frequency or seasonality in the data (i.e. 12 for monthly series)
    :type freq: integer
    :param confidence_level: specified confidence level for the predictive interval, e.g. 0.95
    :type confidence_level: float or list, values between 0 and 1, e.g. [0.9, 0.95]
    :param ts_train: time series in training set (n_timesteps, n_timeseries)
    :type ts_train: numpy array
    :param verbose: print out the metric values
    :type verbose: boolean

    """

    if (not isinstance(metric_list, list)):
        metric_list = [metric_list]

    if (not isinstance(confidence_level, list)):
        confidence_level = [confidence_level]

    metric_value_all_confidence_level = []

    for p in confidence_level:
        alpha = 1.0 - p
        quantiles = [alpha / 2 * 100, (1 - alpha / 2) * 100]

        pred_upper = np.nanpercentile(pred_samples, quantiles[1], axis=0)
        pred_lower = np.nanpercentile(pred_samples, quantiles[0], axis=0)

        metric_value = []
        for i in metric_list:
            if (i == 'msis'):
                name = 'Mean Scaled Interval Score'
                val = msis(pred_upper, pred_lower,
                           data_truth, ts_train, freq, alpha)
                metric_value.append(val)
                if verbose:
                    print('\t {0} at {1}% confidence level: {2:.1f}%'.format(
                        name, int(p*100), val))
            elif (i == 'coverage'):
                name = 'Coverage Percentage'
                val = coverage(pred_upper, pred_lower, data_truth)
                metric_value.append(val)
                if verbose:
                    print('\t {0} at {1}% confidence level: {2:.1f}%'.format(
                        name, int(p*100), val))

        metric_value_all_confidence_level.append(metric_value)

    return metric_value_all_confidence_level
