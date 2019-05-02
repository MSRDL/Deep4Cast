import torch


class LogTransform(object):
    """Natural logarithm of target covariate + `offset`.
    
    .. math:: y_i = log_e ( x_i + \mbox{offset} )
    """

    def __init__(self, offset=0.0, targets=None):
        self.offset = offset
        self.targets = targets

    def do(self, sample):
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
        output['LogTransform_offset'] = self.offset
        output['LogTransform_targets'] = self.targets

        return output
    
    def undo(self, sample):
        output = sample
        X, y = output['X'], output['y']
        offset = output['LogTransform_offset']

        if output['LogTransform_targets']:
            targets = output['LogTransform_targets'][0].tolist()
            X[:, targets, :] = torch.exp(
                X[:, targets, :]) - offset[:, None].float()
            y[:, targets, :] = torch.exp(
                y[:, targets, :]) - offset[:, None].float()
        else:
            X = torch.exp(X) - offset[:, None].float()
            y = torch.exp(y) - offset[:, None].float()
        
        output['X'] = X
        output['y'] = y

        return output


class RemoveLast(object):
    """Subtract last time series points from time series."""

    def __init__(self, targets=None):
        self.targets = targets

    def do(self, sample):
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
        output['RemoveLast_offset'] = offset
        output['RemoveLast_targets'] = self.targets

        return output
    
    def undo(self, sample):
        output = sample
        X, y = output['X'], output['y']
        offset = output['RemoveLast_offset']

        if output['RemoveLast_targets']:
            targets = output['RemoveLast_targets'][0].tolist()
            X[:, targets, :] = X[:, targets, :] + offset[:, None].float()
            y[:, targets, :] = y[:, targets, :] + offset[:, None].float()
        else:
            X += offset[:, None].float()
            y += offset[:, None].float()

        output['X'] = X
        output['y'] = y

        return output


class Tensorize(object):
    """Convert `ndarrays` to Tensors."""

    def __init__(self, device='cpu'):
        self.device = torch.device(device)

    def do(self, sample):
        output = sample
        X, y = sample['X'], sample['y']

        output['X'] = torch.tensor(X, device=self.device).float()
        output['y'] = torch.tensor(y, device=self.device).float()

        return output

    def undo(self, sample):
        return sample


class Target(object):
    """Retain only target covariates for output."""

    def __init__(self, targets):
        self.targets = targets

    def do(self, sample):
        output = sample.copy()
        y = sample['y']

        # Targetize
        output['y'] = y[self.targets, :]
        
        return output

    def undo(self, sample):
        return sample
