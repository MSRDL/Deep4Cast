import numpy as np


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
