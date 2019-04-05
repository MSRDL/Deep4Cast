import numpy as np
import torch


class LogTransform(object):
    """Returns the natural logarithm of a target covariate."""

    def __init__(self, offset=0.0, targets=None):
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
    """Subtract last point from time series."""

    def __init__(self, targets=None):
        self.targets = targets

    def __call__(self, sample):
        output = sample
        X, y = sample['X'], sample['y']

        # Remove offset from X and y
        if self.targets:
            offset = X[self.targets, -1]
            X[self.targets, :] = X[self.targets, :] - offset[:, None]
            y[self.targets, :] = y[self.targets, :] - offset[:, None]
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
    """Standardize time series by subtracting the mean and dividing by the
        standard deviation.
    """

    def __init__(self, targets=None):
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

