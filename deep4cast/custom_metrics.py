import numpy as np


def corr(data_samples, data_truth, agg=None, **kwargs):
    """Computes the empirical correlation betnween actuals and predictions
    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    :param agg: function
    :type agg: aggregator function

    """
    agg = np.median if not agg else agg
    data = agg(data_samples, axis=0)

    return np.round(np.corrcoeff(data, data_truth, rowvar=False), 3)


def mae(data_samples, data_truth, agg=None, **kwargs):
    """Computes mean absolute error (MAE)
    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    :param agg: function
    :type agg: aggregator function

    """
    agg = np.median if not agg else agg
    data = agg(data_samples, axis=0)

    return np.round(np.mean(np.abs(data - data_truth)), 3)


def mape(data_samples, data_truth, agg=None, **kwargs):
    """Computes mean absolute percentage error (MAPE)
    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    :param agg: function
    :type agg: aggregator function

    """
    agg = np.median if not agg else agg
    data = agg(data_samples, axis=0)
    norm = np.abs(data_truth)

    return np.round(np.mean(np.abs(data - data_truth) / norm) * 100.0, 3)


def mase(data_samples, data_truth, data_insample, frequencies, agg=None, **kwargs):
    """Computes mean absolute scaled error (MASE)
    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    :param data_insample: Insample time series values
    :type data_insample: numpy array
    :param frequencies: Frequencies
    :type frequencies: numpy array
    :param agg: function
    :type agg: aggregator function

    """
    agg = np.median if not agg else agg
    data = agg(data_samples, axis=0)

    # Build mean absolute error
    err = np.mean(np.abs(data - data_truth), axis=(1, 2))

    # Build naive absolute error
    t_in = data.shape[-1]
    err_naive = []
    for ts, freq in zip(data_insample, frequencies):
        ts = ts[:, -t_in:]
        naive_forecast = ts[:, :t_in-freq]
        naive_target = ts[:, freq:]
        err_naive.append(np.mean(np.abs(naive_target - naive_forecast)))
    err_naive = np.array(err_naive)

    return np.mean(err[err_naive > 0] / err_naive[err_naive > 0])


def smape(data_samples, data_truth, agg=None, **kwargs):
    """Computes symmetric mean absolute percentage error (SMAPE) on the mean
    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    :param agg: function
    :type agg: aggregator function

    """
    agg = np.median if not agg else agg
    data = agg(data_samples, axis=0)

    eps = 1e-16  # Need to make sure that denominator is not zero
    norm = 0.5 * (np.abs(data) + np.abs(data_truth)) + eps

    return np.round(np.mean(np.abs(data - data_truth) / norm) * 100.0, 3)


def mse(data_samples, data_truth, agg=None, **kwargs):
    """Computes mean squared error (MSE)
    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    :param agg: function
    :type agg: aggregator function

    """
    agg = np.median if not agg else agg
    data = agg(data_samples, axis=0)

    return np.round(np.mean(np.square((data - data_truth))), 3)


def rmse(data_samples, data_truth, agg=None, **kwargs):
    """Computes root-mean squared error (RMSE)
    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    :param agg: function
    :type agg: aggregator function

    """
    agg = np.median if not agg else agg
    data = agg(data_samples, axis=0)

    return np.round(np.sqrt(mse(data, data_truth)), 3)


def rse(data_samples, data_truth, agg=None, **kwargs):
    """Computes root relative squared error (RSE)
    :param data: Predicted time series values (n_timesteps, n_timeseries)
    :type data: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    :param agg: function
    :type agg: aggregator function

    """
    agg = np.median if not agg else agg
    data = agg(data_samples, axis=0)
    norm = np.sqrt(np.sum(data_truth - np.mean(data_truth, axis=0)))

    return np.round(np.sqrt(np.sum(np.square(data - data_truth))) / norm, 3)


def coverage(data_samples, data_truth, percentiles=None, **kwargs):
    """Computes coverage rates of the prediction interval.
    :param data_samples: Samples of time series values
    :type data_samples: numpy array
    :param data_truth: Ground truth time series values
    :type data_truth: numpy array
    :param percentiles: Percentiles to compute coverage for
    :type percentiles: list

    """
    if percentiles is None:
        percentiles = [0.5, 2.5, 5, 25, 50, 75, 95, 97.5, 99.5]

    data_perc = np.percentile(data_samples, q=percentiles, axis=0)
    coverage_percentages = []
    for perc in data_perc:
        coverage_percentages.append(
            np.round(np.mean(data_truth <= perc) * 100.0, 3)
        )

    return coverage_percentages


def pinball_loss(data_samples, data_truth, percentiles=None, **kwargs):
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

    num_steps = data_samples.shape[2]

    # Calculate percentiles
    data_perc = np.percentile(data_samples, q=percentiles, axis=0)

    # Calculate mean pinball loss
    total = 0
    for perc, q in zip(data_perc, percentiles):
        # Calculate upper and lower branch of pinball loss
        upper = data_truth - perc
        lower = perc - data_truth
        upper = np.sum(q / 100.0 * upper[upper >= 0])
        lower = np.sum((1 - q / 100.0) * lower[lower > 0])
        total += (upper + lower) / num_steps

    # Add overall mean pinball loss
    return np.round(total / len(percentiles), 3)

