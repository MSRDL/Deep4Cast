import torch
import numpy as np

from abc import ABC, abstractmethod

from torch.distributions import Normal, Geometric

class ProductDistribution(object):

    def __init__(self, loc, scale, logits):
        self.normal = Normal(loc, scale)
        self.geometric = Geometric(logits)

    def log_prob(self, targets):
        n_targets = targets.shape[1]
        logp_normal = self.normal.log_prob(targets[:,:n_targets//2,:]).sum()
        logp_geometric = self.geometric.log_prob(targets[:,n_targets//2:,:]).sum()

        return logp_normal + logp_geometric

    def sample(self, model_output, n_samples=1) -> torch.Tensor:
        samples = self.dist(*model_output).sample((n_samples,)).cpu()

        return samples

