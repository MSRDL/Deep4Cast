# -*- coding: utf-8 -*-
import torch
import numpy as np

from torch.utils.data import Sampler, Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    """Class to handle time series datasets."""

    def __init__(self, 
            time_series, 
            lookback, 
            horizon, 
            step=1, 
            transform=None, 
            static_covs=None, 
            thinning=1.0):
        """Initialize variables."""
        self.time_series = time_series
        self.lookback = lookback
        self.horizon = horizon
        self.step = step
        self.transform = transform
        self.static_covs = static_covs
        self.thinning = thinning

        # We need to identify each training example in the
        # time series data set, so we assign an ID
        last_id = 0
        n_dropped = 0
        self.sample_ids = {}
        for i, ts in enumerate(self.time_series):
            num_examples = (ts.shape[-1] - self.lookback - self.horizon + self.step) // self.step
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
        return int((max(self.sample_ids.keys()) + 1)*self.thinning)

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

        # Static covariates can be attached
        if self.static_covs is not None:
            sample['X_stat'] = self.static_covs[ts_id]

        return sample


class LogTransform(object):
    """Takes the log of a target covariate."""

    def __init__(self, offset=0.0, targets=None):
        """Initialize variables."""
        self.offset = offset
        self.targets = targets

    def __call__(self, sample):
        output = sample
        X = sample['X']
        y = sample['y']

        # Remove offset from X and y
        if self.targets:
            X[self.targets, :] = torch.log(self.offset + X[self.targets, :]) 
            y[self.targets, :] = torch.log(self.offset + y[self.targets, :]) 
        else:
            X = torch.log(self.offset + X) 
            y = torch.log(self.offset + y) 

        output['X'] = X
        output['y'] = y
        output[self.__class__.__name__ + '_offset'] = self.offset

        return output


class RemoveLast(object):
    """Remove last time series points from time series."""

    def __init__(self, targets=None):
        """Initialize variables."""
        self.targets = targets

    def __call__(self, sample):
        output = sample
        X, y = sample['X'], sample['y']

        # Remove offset from X and y
        if self.targets:
            offset = X[self.targets, -1]
            X[self.targets, :] = X[self.targets, :] - offset[:, None]
            y[self.targets, :]  = y[self.targets, :] - offset[:, None]
        else:
            offset = X[:, -1]
            X = X - offset[:, None]
            y = y - offset[:, None]

        output['X'] = X
        output['y'] = y
        output[self.__class__.__name__ + '_offset'] = offset
        output[self.__class__.__name__ + '_targets'] = self.targets
        
        return output


class Standardize(object):
    """Standardize time series."""

    def __init__(self, targets=None):
        """Initialize variables."""
        self.targets = targets

    def __call__(self, sample):
        output = sample.copy()
        X, y = sample['X'], sample['y']

        # Remove mean from X and y and rescale by standard deviation
        if self.targets:
            mean = X[self.targets, :].mean(dim=1)
            std = X[self.targets, :].std(dim=1)
            X[self.targets, :] -= mean[:, None]
            X[self.targets, :] /= std[:, None]
            y[self.targets, :] -= mean[:, None]
            y[self.targets, :] /= std[:, None]
        else:
            mean = X.mean(dim=1)
            std = X.std(dim=1)
            X -= mean[:, None]
            X /= std[:, None]
            y -= mean[:, None]
            y /= std[:, None]

        output['X'] = X
        output['y'] = y
        output[self.__class__.__name__ + '_mean'] = mean
        output[self.__class__.__name__ + '_std'] = std
        output[self.__class__.__name__ + '_targets'] = self.targets
        
        return output


class Tensorize(object):
    """Convert ndarrays to Tensors."""

    def __init__(self, device='cpu'):
        """Initialize variables."""
        self.device = torch.device(device)

    def __call__(self, sample):
        output = sample.copy()
        X, y = sample['X'], sample['y']

        output['X'] = torch.tensor(X, device=self.device).float()
        output['y'] = torch.tensor(y, device=self.device).float()

        return output


class Target(object):
    """Only keep target covariates for output."""

    def __init__(self, targets):
        self.targets = targets

    def __call__(self, sample):
        output = sample.copy()
        y = sample['y']

        # Targetize
        output['y'] = y[self.targets, :]
        
        return output

