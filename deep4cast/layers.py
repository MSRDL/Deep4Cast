# -*- coding: utf-8 -*-
"""Custom layers module."""

from keras.layers import Dropout
from keras import backend as K
from keras.legacy import interfaces


class MCDropout(Dropout):
    """Applies Dropout to the input, even at prediction time.
    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.
    # Arguments
        rate: float between 0 and 1. Fraction of the input units to drop.
        noise_shape: 1D integer tensor representing the shape of the
            binary dropout mask that will be multiplied with the input.
            For instance, if your inputs have shape
            `(batch_size, timesteps, features)` and
            you want the dropout mask to be the same for all timesteps,
            you can use `noise_shape=(batch_size, 1, features)`.
        seed: A Python integer to use as random seed.
    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from
        Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/
        srivastava14a.pdf)
        - [MCDropout: Dropout as a Bayesian Approximation: Representing Model
        Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142)
    """

    @interfaces.legacy_dropout_support
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        """Initialize variables."""
        super(MCDropout, self).__init__(
            rate,
            noise_shape=None,
            seed=None,
            **kwargs
        )

    def call(self, inputs, training=None):
        """Return dropped weights when called."""
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)
            output = K.dropout(inputs, self.rate, noise_shape, seed=self.seed)

            return output

        return inputs
