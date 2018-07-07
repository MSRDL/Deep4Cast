# -*- coding: utf-8 -*-
"""Custom layers module."""
import numpy as np

from keras.layers import Layer, Dropout
from keras import backend as K
from keras.legacy import interfaces
from keras import initializers
from keras.engine import InputSpec


class ConcreteDropout(Layer):
    @interfaces.legacy_dropout_support
    def __init__(self, regularizer=1e-5, init_range=(0.1, 0.3), **kwargs):
        super(ConcreteDropout, self).__init__(**kwargs)
        self.regularizer = regularizer
        self.init_range = init_range
        self.supports_masking = True

        # Dropout regularizer parameters
        self.p_logit = None
        self.p = None
        self.init_min = np.log(init_range[0]) - np.log(1. - init_range[0])
        self.init_max = np.log(init_range[1]) - np.log(1. - init_range[1])

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)

        # This is very weird. We must call super before we add new losses
        # super(ConcreteDropout, self).build(input_shape)

        # Initialize the dropout probability
        self.p_logit = self.add_weight(
            name='p_logit',
            shape=(1,),
            initializer=initializers.RandomUniform(
                self.init_min,
                self.init_max
            ),
            trainable=True
        )
        self.p = K.sigmoid(self.p_logit[0])

        # Initialize the dropout regularizer
        input_dim = np.prod(input_shape[1:])  # Drop only last two dims
        regularizer = self.p * K.log(self.p)
        regularizer += (1. - self.p) * K.log(1. - self.p)
        regularizer *= self.regularizer * input_dim
        self.add_loss(regularizer)

    def concrete_dropout(self, inputs):
        '''
        Applies approx. dropping to inputs such that gradients can be
        propagated.

        '''
        # Parameters for concrete distribution
        eps = K.cast_to_floatx(K.epsilon())
        temp = 0.1

        # Use concrete distribution to calculate approx. dropout mask
        unif_noise = K.random_uniform(shape=K.shape(inputs))
        drop_prob = (
            K.log(self.p + eps) -
            K.log(1. - self.p + eps) +
            K.log(unif_noise + eps) -
            K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob
        retain_prob = 1. - self.p

        # Approximately drop inputs
        inputs *= random_tensor
        inputs /= retain_prob

        return inputs

    def call(self, inputs, training=None):
        return self.concrete_dropout(inputs)

    def get_config(self):
        config = {'regularizer': self.regularizer,
                  'init_range': self.init_range}
        base_config = super(ConcreteDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


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


class TemporalAttention(Layer):
    """Applies simple static attention mechanism to tensor of shape
    (batch_size, time_steps, n_units). This can be applied for long-term
    prediction tasks by finding the the past state vectors of LSTMs/GRUs that
    are most relevant for the next prediction step.
    # References
        - [Modeling Long- and Short-Term Temporal Patterns with Deep Neural
        Networks](https://arxiv.org/abs/1703.07015)
    """

    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable kernel and bias for output.
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[1] + 1, input_shape[2]),
            initializer='uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            name='kernel',
            shape=(1, input_shape[2]),
            initializer='uniform',
            trainable=True
        )
        super(TemporalAttention, self).build(input_shape)

    def call(self, x):
        # Normalize input for calculating cosine distance
        x_norm = K.l2_normalize(x, axis=-1)

        # Calculate weights for each time step as cosine distance between
        # every time step's state vectors and last time step.
        weights = K.batch_dot(x_norm, x[:, -1, :])
        weights = K.expand_dims(weights)

        # Calculate weighted input for each time step and concatenate
        # with most recent state vector.
        c = x * weights
        c = K.concatenate([c, K.expand_dims(x[:, -1, :], axis=1)], axis=1)

        return K.sum(c * self.kernel, axis=1) + self.bias

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


class AutoRegression(Layer):
    """Applies simple autoregression to input time series.
    # References
        - [Modeling Long- and Short-Term Temporal Patterns with Deep Neural
        Networks](https://arxiv.org/abs/1703.07015)
    """

    def __init__(self, **kwargs):
        super(AutoRegression, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable kernel and bias for output.
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[1], input_shape[2]),
            initializer='uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            name='kernel',
            shape=(1, input_shape[2]),
            initializer='uniform',
            trainable=True
        )
        super(AutoRegression, self).build(input_shape)

    def call(self, x):
        return K.sum(x * self.kernel, axis=1) + self.bias

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
