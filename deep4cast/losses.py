import torch
import numpy as np

from abc import ABC, abstractmethod


class Loss(ABC):    
    """Abstract base class for handling general loss functions for forecasters."""

    @abstractmethod
    def evaluate(self):
        """Returns the negative log-likelihood of
        the distribution that defines this loss function.

        """
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        """Returns a tensor of sampled predictions."""
        raise NotImplementedError


class LikelihoodLoss(Loss):
    """Handles univariate likelihood losses.

    Arguments:
        * dist (``torch.distributions``): Instance of PyTorch `distribution <https://pytorch.org/docs/stable/distributions.html>`_.

    """

    def __init__(self, dist):
        """Initialize variables."""
        self.dist = dist

    def evaluate(self, model_output, targets):
        """Returns the negative log-likelihood of
        the distribution that defines this loss function

        Arguments:
            * model_output (``list``): List of model outputs parameterizing the distribution.
            * targets (``torch.Tensor``): Tensor that contains the observations to be learned from.

        """
        neg_ll = -self.dist(*model_output).log_prob(targets).sum()

        return neg_ll

    def sample(self, model_output, n_samples=1) -> torch.Tensor:
        """Returns a tensor of sampled predictions.
        
        Arguments:
            * model_output (``list``): List of model outputs parameterizing the distribution.
            * n_samples (``int``): number of samples to be drawn.
        
        """
        samples = self.dist(*model_output).sample((n_samples,)).cpu()

        return samples

