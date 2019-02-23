# -*- coding: utf-8 -*-
"""Custom layers module."""
import torch
import numpy as np


class ConcreteDropout(torch.nn.Module):
    """Applies Dropout to the input, even at prediction time.

    Dropout consists in randomly setting a fraction `rate` of input units to 0
    at each update during training time, which helps prevent overfitting. At
    prediction time the units are then also dropped out with the same fraction.
    This generates samples from an approximate posterior predictive
    distribution. Unlike in MCDropout, in Concrete Dropout the dropout rates
    are learned from the data.

    References
        - [Dropout: A Simple Way to Prevent Neural Networks from
        Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/
        srivastava14a.pdf)
        - [MCDropout: Dropout as a Bayesian Approximation: Representing Model
        Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142)
        - [Concrete Dropout](https://papers.nips.cc/
        paper/6949-concrete-dropout.pdf)

    """

    def __init__(self,
                 dropout_regularizer=1e-5,
                 init_range=(0.1, 0.3),
                 channel_wise=False):
        super(ConcreteDropout, self).__init__()

        self.dropout_regularizer = dropout_regularizer
        self.init_range = init_range
        self.channel_wise = channel_wise

        # Initialize dropout probability
        init_min = np.log(init_range[0]) - np.log(1. - init_range[0])
        init_max = np.log(init_range[1]) - np.log(1. - init_range[1])
        self.p_logit = torch.nn.Parameter(
            torch.empty(1).uniform_(init_min, init_max))

    def forward(self, x):
        # Get the dropout probability
        p = torch.sigmoid(self.p_logit)

        # Apply Concrete Dropout to input
        out = self._concrete_dropout(x, p)

        # Regularization term for dropout parameters
        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)

        # The size of the dropout regularization depends on the input size
        # The input dim used to set up the dropout probability reg,
        # depends in whether the dropout mask is constant accross time or not
        if self.channel_wise:
            input_dim = x.shape[1]  # Dropout only applied to channel dimension
        else:
            # Dropout applied to all dimensions
            input_dim = np.prod(x.shape[1:])
        dropout_regularizer *= self.dropout_regularizer * input_dim

        return out, dropout_regularizer.mean()

    def _concrete_dropout(self, x, p):
        eps = 1e-7
        temp = 0.1

        # Apply Concrete dropout channel wise or across all input
        if self.channel_wise:
            unif_noise = torch.rand_like(x[:, :, [0]])
        else:
            unif_noise = torch.rand_like(x)

        drop_prob = (torch.log(p + eps)
                     - torch.log(1 - p + eps)
                     + torch.log(unif_noise + eps)
                     - torch.log(1 - unif_noise + eps))
        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob

        # Need to make sure we have the right shape for the Dropout mask
        if self.channel_wise:
            random_tensor = random_tensor.repeat([1, 1, x.shape[2]])

        # Now drop weights
        retain_prob = 1 - p
        x = torch.mul(x, random_tensor)
        x /= retain_prob

        return x

