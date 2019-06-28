import numpy as np
from torch.utils.data import Dataset

from deep4cast import transforms


class TimeSeriesDataset(Dataset):
    """Takes a list of time series and provides access to windowed subseries for
    training.

    Arguments:
        * time_series (list): List of time series ``numpy`` arrays.
        * lookback (int): Number of time steps used as input for forecasting.
        * horizon (int): Number of time steps to forecast.
        * step (int): Time step size between consecutive examples.
        * transform (``transforms.Compose``): Specific transformations to apply to time series examples.
        * static_covs (list): Static covariates for each item in ``time_series`` list.

    """
    def __init__(self, 
                 time_series,
                 lookback,
                 horizon,
                 step, 
                 transform,
                 static_covs=None):
        self.time_series = time_series
        self.lookback = lookback
        self.horizon = horizon
        self.step = step
        self.transform = transform
        self.static_covs = static_covs

        # Slice each time series into examples, assigning IDs to each
        last_id = 0
        n_dropped = 0
        self.example_ids = {}
        for i, ts in enumerate(self.time_series):
            num_examples = (ts.shape[-1] - self.lookback - self.horizon + self.step) // self.step
            # Time series shorter than the forecast horizon need to be dropped.
            if ts.shape[-1] < self.horizon:
                n_dropped += 1
                continue
            # For short time series zero pad the input
            if ts.shape[-1] < self.lookback + self.horizon:
                num_examples = 1
            for j in range(num_examples):
                self.example_ids[last_id + j] = (i, j * self.step)
            last_id += num_examples

        # Inform user about time series that were too short
        if n_dropped > 0:
            print("Dropped {}/{} time series due to length.".format(
                    n_dropped, len(self.time_series)
                    )
                 )

        # Store the number of training examples
        self._len = self.example_ids.__len__()

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # Get time series
        ts_id, lookback_id = self.example_ids[idx]
        ts = self.time_series[ts_id]

        # Prepare input and target. Zero pad if necessary.
        if ts.shape[-1] < self.lookback + self.horizon:
            # If the time series is too short, we zero pad
            X = ts[:, :-self.horizon]
            X = np.pad(
                X, 
                pad_width=((0, 0), (self.lookback - X.shape[-1], 0)), 
                mode='constant', 
                constant_values=0
            )
            y = ts[:, -self.horizon:]
        else:
            X = ts[:, lookback_id:lookback_id + self.lookback]
            y = ts[:, lookback_id + self.lookback:lookback_id + self.lookback + self.horizon]

        # Create the input and output for the sample
        sample = {'X': X, 'y': y}
        sample = self.transform(sample)

        # Static covariates can be attached
        if self.static_covs is not None:
            sample['X_stat'] = self.static_covs[ts_id]

        return sample
