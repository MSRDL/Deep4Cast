from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """Take a list of time series and provides access to windowed subseries for
    training.
    
    :param time_series: List of time series arrays.
    :param lookback: Length of time window used as input for forecasting.
    :param horizon: Number of time steps to forecast.
    :param step: Time step size between consecutive examples.
    :param dropout_regularizer: Generally needs to be set to 2 / N, where N is
        the number of training examples.
    :param init_range: Initial range for dropout probabilities.
    :param channel_wise: Determines if dropout is appplied accross all input or
        across channels .
    """
    def __init__(self, 
            time_series: list, 
            lookback: int, 
            horizon: int, 
            step=1, 
            transform=None, 
            static_covs=None):
        self.time_series = time_series
        self.lookback = lookback
        self.horizon = horizon
        self.step = step
        self.transform = transform
        self.static_covs = static_covs

        # We need to identify each training example in the time series data set, so we assign an ID
        last_id = 0
        n_dropped = 0
        self.sample_ids = {}
        for i, ts in enumerate(self.time_series):
            num_examples = (ts.shape[-1] - self.lookback - self.horizon + self.step) // self.step

            # Time series that are too short can be zero-padded but time series that are
            # shorter than the forecast horizon need to be dropped.
            if ts.shape[-1] < self.lookback + self.horizon:
                num_examples = 1 # If the time series is too short, we will zero pad the input
            if ts.shape[-1] < self.horizon:
                n_dropped += 1
                continue
            for j in range(num_examples):
                self.sample_ids[last_id + j] = (i, j*self.step)
            last_id += num_examples

        # Inform user about time series that were too short
        if n_dropped:
            print(
                "Dropped {}/{} time series due to length.".format(n_dropped, len(self.time_series))
            )
        
    def __len__(self):
        # Returns the number of training examples
        return int(max(self.sample_ids.keys()) + 1)

    def __getitem__(self, idx):
        # Get time series
        ts_id, lookback_id = self.sample_ids[idx]
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
        if self.transform:
            sample = self.transform(sample)

        # Static covariates can be attached if available
        if self.static_covs is not None:
            sample['X_stat'] = self.static_covs[ts_id]

        return sample