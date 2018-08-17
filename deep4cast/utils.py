import numpy as np


def check_data_format(data, horizon):
    """Raise error if data has incorrect format."""
    # Check if data has any NaNs.
    if np.isnan([np.isnan(x).any() for x in data]).any():
        raise ValueError('data should not contain NaNs.')

    # Check if data is long enough for forecasting horizon.
    if np.array([len(x) <= horizon for x in data]).any():
        raise ValueError('Time series must be longer than horizon.')


def sequentialize(data, lag, horizon, targets=None):
    """Sequentialize time series array.
    Create two numpy arrays, one for the windowed input time series X
    and one for the corresponding output values that need to be
    predicted.
    """

    # Sequentialize the dataset, i.e., split it into shorter windowed
    # sequences.
    X, y = [], []
    for time_series in data:
        # Making sure the time_series dataset is in correct format
        time_series = np.atleast_2d(time_series)

        # Need the number of time steps per window and the number of
        # covariates
        n_time_steps, n_vars = time_series.shape

        # No build the data structure
        for j in range(n_time_steps - lag + 1):
            lag_ts = time_series[j:j + lag]
            forecast_ts = time_series[j + lag:j + lag + horizon]
            if len(forecast_ts) < horizon:
                forecast_ts = np.ones(shape=(horizon, n_vars)) * np.nan
            X.append(lag_ts)
            if targets:
                y.append(forecast_ts[:, targets])
            else:
                y.append(forecast_ts)

    if not X or not y:
        raise ValueError(
            'Time series is too short for lag and/or horizon. lag {} + horizon {} > n_time_steps {}.'.format(
                lag, horizon,
                n_time_steps
            )
        )

    # Make sure we output numpy arrays.
    X = np.array(X)
    y = np.array(y)

    # Remove NaNs that occur during windowing
    X = X[~np.isnan(y)[:, 0, 0]]
    y = y[~np.isnan(y)[:, 0, 0]]

    return X, y


def generate_folds(self, data, lag, n_folds, test_fraction):
    """Yield a data fold."""
    # Find the maximum length of all example time series in the dataset.
    data_length = []
    for time_series in data:
        data_length.append(len(time_series))
    data_length = max(data_length)
    test_length = int(data_length * test_fraction)
    train_length = data_length - n_folds * test_length

    # Loop over number of folds to generate folds for cross-validation
    # but make sure that the folds do not overlap.
    for i in range(n_folds):
        data_train, data_val = [], []
        for time_series in data:
            train_ind = np.arange(
                -(i + 1) * test_length - train_length,
                -(i + 1) * test_length
            )
            test_ind = np.arange(
                -(i + 1) * test_length - lag,
                -i * test_length
            )
            data_train.append(time_series[train_ind, :])
            data_val.append(time_series[test_ind, :])
        data_train = np.array(data_train)
        data_val = np.array(data_val)

        yield data_train, data_val


def transform(X, func, targets=None):
    if targets is None:
        return func(X)
    else:
        _X = X.copy()
        _X[:,:,targets] = func(_X[:,:,targets])
        return _X


class VectorScaler():
    """Defines a VectorScaler."""

    def __init__(self, targets=None):
        self.targets = targets
        self.mean = None
        self.std = None

    def fit(self, X, y=None, **kwargs):
        """Fit the scaler."""
        if self.targets is None:
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
        else:
            # Need to concatenate mean with zeros and stds with ones for
            # categorical targets
            mean = np.mean(X[:, :, self.targets], axis=0)
            std = np.std(X[:, :, self.targets], axis=0)
            cat_shape = (X.shape[1],
                         X.shape[2] - len(self.targets))
            zeros = np.zeros(shape=cat_shape)
            ones = np.ones(shape=cat_shape)
            mean = np.concatenate((mean, zeros), axis=1)
            std = np.concatenate((std, ones), axis=1)

        self.mean = mean
        self.std = std

    def transform(self, X, **kwargs):
        return (X - self.mean) / self.std

    def inverse_transform(self, X, **kwargs):
        return X * self.std + self.mean
