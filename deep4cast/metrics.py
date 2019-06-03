import numpy as np
import warnings


def mae(data_samples:np.array, data_truth:np.array, agg=None) -> np.array:
    """Computes mean absolute error (MAE)

    Arguments:
        * data_samples: sampled predictions (n_samples, n_timeseries, n_variables, n_timesteps)
        * data_truth: ground truth time series values (n_timeseries, n_variables, n_timesteps)
        * agg: sample aggregation function

    """
    if data_samples.shape[1:] != data_truth.shape:
        raise ValueError('Last three dimensions of data_samples and data_truth need to be compatible')
    agg = np.median if not agg else agg

    # Aggregate over samples
    data = agg(data_samples, axis=0)

    return np.mean(np.abs(data - data_truth), axis=(1, 2))


def mape(data_samples:np.array, data_truth:np.array, agg=None) -> np.array:
    """Computes mean absolute percentage error (MAPE)

    Arguments:
        * data_samples: sampled predictions (n_samples, n_timeseries, n_variables, n_timesteps)
        * data_truth: ground truth time series values (n_timeseries, n_variables, n_timesteps)
        * agg: sample aggregation function

    """
    if data_samples.shape[1:] != data_truth.shape:
        raise ValueError('Last three dimensions of data_samples and data_truth need to be compatible')
    agg = np.median if not agg else agg

    # Aggregate over samples
    data = agg(data_samples, axis=0)
    
    norm = np.abs(data_truth)

    return np.mean(np.abs(data - data_truth) / norm, axis=(1, 2)) * 100.0


def mase(data_samples:np.array, 
         data_truth:np.array, 
         data_insample:np.array, 
         frequencies:list, 
         agg=None) -> np.array:
    """Computes mean absolute scaled error (MASE) as in the `M4 competition
    <https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf>`_.

    Arguments:
        * data_samples: sampled predictions (n_samples, n_timeseries, n_variables, n_timesteps)
        * data_truth: ground truth time series values (n_timeseries, n_variables, n_timesteps)
        * data_insample: in-sample time series data (n_timeseries, n_variables, n_timesteps)
        * frequencies: frequencies to be used when calculating the naive forecast
        * agg: sample aggregation function

    """
    if data_samples.shape[1:] != data_truth.shape:
        raise ValueError('Last three dimensions of data_samples and data_truth need to be compatible')
    agg = np.median if not agg else agg

    # Calculate mean absolute for forecast and naive forecast per time series
    errs, naive_errs = [], []
    for i in range(data_samples.shape[1]):
        ts_sample = data_samples[:, i]
        ts_truth = data_truth[i]
        ts = data_insample[i]
        freq = int(frequencies[i])

        data = agg(ts_sample, axis=0)

        # Build mean absolute error
        err = np.mean(np.abs(data - ts_truth))

        # naive forecast is calculated using insample
        t_in = ts.shape[-1]
        naive_forecast = ts[:, :t_in-freq]
        naive_target = ts[:, freq:]
        err_naive = np.mean(np.abs(naive_target - naive_forecast))

        errs.append(err)
        naive_errs.append(err_naive)
    
    errs = np.array(errs)
    naive_errs = np.array(naive_errs)

    return errs / naive_errs


def smape(data_samples:np.array, data_truth:np.array, agg=None) -> np.array:
    """Computes symmetric mean absolute percentage error (SMAPE) on the mean
    
    Arguments:
        * data_samples: sampled predictions (n_samples, n_timeseries, n_variables, n_timesteps)
        * data_truth: ground truth time series values (n_timeseries, n_variables, n_timesteps)
        * agg: sample aggregation function

    """
    if data_samples.shape[1:] != data_truth.shape:
        raise ValueError('Last three dimensions of data_samples and data_truth need to be compatible')
    agg = np.median if not agg else agg

    # Aggregate over samples
    data = agg(data_samples, axis=0)

    eps = 1e-16  # Need to make sure that denominator is not zero
    norm = 0.5 * (np.abs(data) + np.abs(data_truth)) + eps

    return np.mean(np.abs(data - data_truth) / norm, axis=(1, 2)) * 100


def mse(data_samples:np.array, data_truth:np.array, agg=None) -> np.array:
    """Computes mean squared error (MSE)
    
    Arguments:
        * data_samples: sampled predictions (n_samples, n_timeseries, n_variables, n_timesteps)
        * data_truth: ground truth time series values (n_timeseries, n_variables, n_timesteps)
        * agg: sample aggregation function

    """
    if data_samples.shape[1:] != data_truth.shape:
        raise ValueError('Last three dimensions of data_samples and data_truth need to be compatible')
    agg = np.median if not agg else agg

    # Aggregate over samples
    data = agg(data_samples, axis=0)

    return np.mean(np.square((data - data_truth)), axis=(1, 2))


def rmse(data_samples:np.array, data_truth:np.array, agg=None) -> np.array:
    """Computes mean squared error (RMSE)
    
    Arguments:
        * data_samples: sampled predictions (n_samples, n_timeseries, n_variables, n_timesteps)
        * data_truth: ground truth time series values (n_timeseries, n_variables, n_timesteps)
        * agg: sample aggregation function

    """
    if data_samples.shape[1:] != data_truth.shape:
        raise ValueError('Last three dimensions of data_samples and data_truth need to be compatible')
    agg = np.median if not agg else agg

    # Aggregate over samples
    data = agg(data_samples, axis=0)

    return np.sqrt(mse(data, data_truth))


def coverage(data_samples:np.array, data_truth:np.array, percentiles=None) -> list:
    """Computes coverage rates of the prediction interval.

    Arguments:
        * data_samples: sampled predictions (n_samples, n_timeseries, n_variables, n_timesteps)
        * data_truth: ground truth time series values (n_timeseries, n_variables, n_timesteps)
        * percentiles: list of percentiles to calculate coverage for

    """
    if data_samples.shape[1:] != data_truth.shape:
        raise ValueError('Last three dimensions of data_samples and data_truth need to be compatible')
    if percentiles is None:
        percentiles = [0.5, 2.5, 5, 25, 50, 75, 95, 97.5, 99.5]

    data_perc = np.percentile(data_samples, q=percentiles, axis=0)
    coverage_percentages = []
    for perc in data_perc:
        coverage_percentages.append(
            np.round(np.mean(data_truth <= perc) * 100.0, 3)
        )

    return coverage_percentages


def pinball_loss(data_samples:np.array, data_truth:np.array, percentiles=None) -> np.array:
    """Computes pinball loss.

    Arguments:
        * data_samples: sampled predictions (n_samples, n_timeseries, n_variables, n_timesteps)
        * data_truth: ground truth time series values (n_timeseries, n_variables, n_timesteps)
        * percentiles: list of percentiles to calculate coverage for

    """
    if data_samples.shape[1:] != data_truth.shape:
        raise ValueError('Last three dimensions of data_samples and data_truth need to be compatible')
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


def msis(data_samples:np.array, 
         data_truth:np.array, 
         data_insample:np.array, 
         frequencies:list, 
         alpha=0.05) -> np.array:
    """Mean Scaled Interval Score (MSIS) as shown in the `M4 competition 
    <https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf>`_.

    Arguments:
        * data_samples: sampled predictions (n_samples, n_timeseries, n_variables, n_timesteps)
        * data_truth: ground truth time series values (n_timeseries, n_variables, n_timesteps)
        * data_insample: in-sample time series data (n_timeseries, n_variables, n_timesteps)
        * frequencies: frequencies to be used when calculating the naive forecast
        * alpha: significance level
    
    """
    if data_samples.shape[1:] != data_truth.shape:
        raise ValueError('Last three dimensions of data_samples and data_truth need to be compatible')
    lower = (alpha / 2) * 100
    upper = 100 - (alpha / 2) * 100

    # drop individual samples for a given time series where the prediction is
    # not finite
    penalty_us, penalty_ls, scores, seas_diffs = [], [], [], []
    for i in range(data_samples.shape[1]):
        # Set up individual time series
        ts_sample = data_samples[:, i]
        ts_truth = data_truth[i]
        ts = data_insample[i]
        freq = int(frequencies[i])

        mask = np.where(~np.isfinite(ts_sample))[0]
        if mask.shape[0] > 0:
            mask = np.unique(mask)
            warnings.warn('For time series {}, removing {} of {} total samples.'.format(
                i, mask.shape[0], ts_sample.shape[0]))
            ts_sample = np.delete(ts_sample, mask, axis=0)
        
        # Calculate percentiles
        data_perc = np.percentile(ts_sample, q=(lower, upper), axis=0)

        # Penalty is (lower - actual) + (actual - upper)
        penalty_l = data_perc[0] - ts_truth
        penalty_l = np.where(penalty_l > 0, penalty_l, 0)
        penalty_u = ts_truth - data_perc[1]
        penalty_u = np.where(penalty_u > 0, penalty_u, 0)

        penalty_u = (2 / alpha) * np.mean(penalty_u, axis=1)
        penalty_l = (2 / alpha) * np.mean(penalty_l, axis=1)

        # Score is upper - lower
        score = np.mean(data_perc[1] - data_perc[0], axis=1)

        # Naive forecast is calculated using insample data
        t_in = ts.shape[-1]
        ts = ts[-t_in:]
        naive_forecast = ts[:, :t_in-freq]
        naive_target = ts[:, freq:]
        seas_diff = np.mean(np.abs(naive_target - naive_forecast))

        penalty_us.append(penalty_u)
        penalty_ls.append(penalty_l)
        scores.append(score)
        seas_diffs.append(seas_diff)

    penalty_us = np.concatenate(penalty_us)
    penalty_ls = np.concatenate(penalty_ls)
    scores = np.concatenate(scores)
    seas_diffs = np.array(seas_diffs)

    return (scores + penalty_us + penalty_ls) / seas_diffs


def acd(data_samples:np.array, data_truth:np.array, alpha=0.05) -> float:
    """The absolute difference between the coverage of the method and the target (0.95).

    Arguments:
        * data_samples: samples of time series values
        * data_truth: ground truth time series values
        * alpha: percentile to compute coverage difference

    """
    if data_samples.shape[1:] != data_truth.shape:
        raise ValueError('Last three dimensions of data_samples and data_truth need to be compatible')

    alpha = (1 - alpha) * 100
    data_perc = np.percentile(data_samples, q=[alpha], axis=0)
    acd = alpha - np.round(np.mean(data_truth <= data_perc[0]) * 100.0, 3)
    acd = np.abs(acd) / 100

    return acd

