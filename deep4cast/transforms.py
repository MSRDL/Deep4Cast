import torch


class Compose(object):
    r"""Composes several transforms together.
    
    List of transforms must currently begin with ``ToTensor`` and end with
    ``Target``.

    Args:
        * transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.ToTensor(),
        >>>     transforms.LogTransform(targets=[0], offset=1.0),
        >>>     transforms.Target(targets=[0]),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, example):
        for t in self.transforms:
            example = t(example)
        return example

    def untransform(self, example):
        for t in self.transforms[::-1]:
            example = t.untransform(example)
        return example


class LogTransform(object):
    r"""Natural logarithm of target covariate + `offset`.
    
    .. math:: y_i = log_e ( x_i + \mbox{offset} )

    Args:
        * offset (float): amount to add before taking the natural logarithm
        * targets (list): list of indices to transform.

    Example:
        >>> transforms.LogTransform(targets=[0], offset=1.0)
    """

    def __init__(self, targets=None, offset=0.0):
        self.offset = offset
        self.targets = targets

    def __call__(self, sample):
        X = sample['X']
        y = sample['y']

        if self.targets:
            X[self.targets, :] = torch.log(self.offset + X[self.targets, :])
            y[self.targets, :] = torch.log(self.offset + y[self.targets, :])
        else:
            X = torch.log(self.offset + X)
            y = torch.log(self.offset + y)

        sample['X'] = X
        sample['y'] = y

        return sample

    def untransform(self, sample):
        X, y = sample['X'], sample['y']

        # Unpack nested list of forecasting targets.
        Target_targets = [torch.unique(x).tolist()
                          for x in sample['Target_targets']]
        Target_targets = sum(Target_targets, [])

        # If the transform target and forecast target overlap then find the
        # corresponding index in the y array.
        intersect = set(self.targets).intersection(Target_targets)
        indices_y = [i for i, item in enumerate(
            Target_targets) if item in intersect]

        if self.targets:
            X[:, self.targets, :] = \
                torch.exp(X[:, self.targets, :]) - self.offset
        else:
            X = torch.exp(X) - self.offset
            y = torch.exp(y) - self.offset

        # Exponentiate only those forecasting targets where we took the
        # natural log.
        if len(intersect) > 0:
            y[:, indices_y, :] = torch.exp(y[:, indices_y, :]) - self.offset

        sample['X'] = X
        sample['y'] = y

        return sample


class RemoveLast(object):
    r"""Subtract final point in lookback window from all points in example.
    
    Args:
        * targets (list): list of indices to transform.

    Example:
        >>> transforms.RemoveLast(targets=[0])
    """

    def __init__(self, targets=None):
        self.targets = targets

    def __call__(self, sample):
        X, y = sample['X'], sample['y']

        if self.targets:
            offset = X[self.targets, -1]
            X[self.targets, :] = X[self.targets, :] - offset[:, None]
            y[self.targets, :] = y[self.targets, :] - offset[:, None]
        else:
            offset = X[:, -1]
            X = X - offset[:, None]
            y = y - offset[:, None]

        sample['RemoveLast_offset'] = offset

        return sample

    def untransform(self, sample):
        X, y = sample['X'], sample['y']
        offset = sample['RemoveLast_offset']

        # Unpack nested list of forecasting targets.
        Target_targets = \
            [torch.unique(x).tolist() for x in sample['Target_targets']]
        Target_targets = sum(Target_targets, [])

        # If the transform target and forecast target overlap then find the
        # corresponding index in the y array.
        intersect = set(self.targets).intersection(Target_targets)

        if self.targets:
            X[:, self.targets, :] = \
                X[:, self.targets, :] + offset[:, :, None].float()
        else:
            X += offset[:, :, None].float()
            y += offset[:, Target_targets, None].float()

        # Add back to the correct forecasted index the quantity removed
        if len(intersect) > 0:
            indices_o = \
                [i for i, item in enumerate(self.targets) if item in intersect]
            indices_y = \
                [i for i, item in enumerate(
                    Target_targets) if item in intersect]
            y[:, indices_y, :] = \
                y[:, indices_y, :] + offset[:, indices_o, None].float()

        sample['X'] = X
        sample['y'] = y

        return sample


class ToTensor(object):
    r"""Convert ``numpy.ndarrays`` to tensor.
    
    Args:
        * device (str): device on which to load the tensor.

    Example:
        >>> transforms.ToTensor(device='cpu')
    """

    def __init__(self, device='cpu'):
        self.device = torch.device(device)

    def __call__(self, sample):
        sample['X'] = torch.tensor(sample['X'], device=self.device).float()
        sample['y'] = torch.tensor(sample['y'], device=self.device).float()

        return sample

    def untransform(self, sample):
        return sample


class Target(object):
    r"""Retain only target indices for output.

    Args:
        * targets (list): list of indices to retain.

    Example:
        >>> transforms.Target(targets=[0])
    """

    def __init__(self, targets):
        self.targets = targets

    def __call__(self, sample):
        sample['y'] = sample['y'][self.targets, :]
        sample['Target_targets'] = self.targets

        return sample

    def untransform(self, sample):
        return sample


class Standardize(object):
    """Subtract the mean and divide by the standard deviation from the lookback.

    Args:
        * targets (list): list of indices to transform.

    Example:
        >>> transforms.Standardize(targets=[0])
    """

    def __init__(self, targets=None):
        self.targets = targets

    def __call__(self, sample):
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

        sample['X'] = X
        sample['y'] = y
        sample['Standardize_mean'] = mean
        sample['Standardize_std'] = std

        return sample

    def untransform(self, sample):
        X, y = sample['X'], sample['y']

        # Unpack nested list of forecasting targets.
        Target_targets = \
            [torch.unique(x).tolist() for x in sample['Target_targets']]
        Target_targets = sum(Target_targets, [])

        # If the transform target and forecast target overlap then find the
        # corresponding index in the y array.
        intersect = set(self.targets).intersection(Target_targets)

        if self.targets:
            X[:, self.targets, :] = \
                X[:, self.targets, :] * sample['Standardize_std'][:, :, None]
            X[:, self.targets, :] = \
                X[:, self.targets, :] + sample['Standardize_mean'][:, :, None]
        else:
            X = X * sample['Standardize_std']
            X = X + sample['Standardize_mean']
            y = y * sample['Standardize_std'][:, Target_targets, None]
            y = y + sample['Standardize_mean'][:, Target_targets, None]

        # Add back to the correct index the quantity removed
        if len(intersect) > 0:
            # indices for the offset
            indices_o = \
                [i for i, item in enumerate(self.targets) if item in intersect]
            # indices for the target
            indices_y = \
                [i for i, item in enumerate(
                    Target_targets) if item in intersect]
            y[:, indices_y, :] = \
                y[:, indices_y, :] * sample['Standardize_std'][:, indices_o, None]
            y[:, indices_y, :] = \
                y[:, indices_y, :] + sample['Standardize_mean'][:, indices_o, None]

        return sample
