# -*- coding: utf-8 -*-
"""Custom layers module."""
import numpy as np

from keras.layers import Layer
from keras import backend as K
from keras.legacy import interfaces
from keras import initializers
from keras.engine import InputSpec


class ConcreteDropout(Layer):
    """Applies Dropout to the input, even at prediction time.
    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.
    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from
        Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/
        srivastava14a.pdf)
        - [MCDropout: Dropout as a Bayesian Approximation: Representing Model
        Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142)
        - [Concrete Dropout](https://papers.nips.cc/
        paper/6949-concrete-dropout.pdf)
    """
    @interfaces.legacy_dropout_support
    def __init__(self,
                 temporal_dropout=False,
                 regularizer=1e-5,
                 init_range=(0.1, 0.3),
                 **kwargs):
        super(ConcreteDropout, self).__init__(**kwargs)
        self.regularizer = regularizer
        self.init_range = init_range
        self.supports_masking = True
        self.temporal_dropout = temporal_dropout

        # Dropout regularizer parameters
        self.p_logit = None
        self.p = None
        self.init_min = np.log(init_range[0]) - np.log(1. - init_range[0])
        self.init_max = np.log(init_range[1]) - np.log(1. - init_range[1])

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)

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

        # The input dim used to set up the dropout probability regularizer,
        # depends in whether the dropout mask is constant accross time or not
        if self.temporal_dropout:
            input_dim = self.input_spec.shape[-1]  # Drop only last dim
        else:
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

        # The dropout mask needs to be held constant, when Dropout is applied
        # to temporal data (i.e., same mask for each time step)
        if self.temporal_dropout:
            shape = (K.shape(inputs)[0], 1, K.shape(inputs)[2])
        else:
            shape = K.shape(inputs)

        # Use concrete distribution to calculate approx. dropout mask
        unif_noise = K.random_uniform(shape=shape)
        drop_prob = (
            K.log(self.p + eps) -
            K.log(1. - self.p + eps) +
            K.log(unif_noise + eps) -
            K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        retain_prob = 1. - self.p

        # The dropout mask needs to be held constant, when Dropout is applied
        # to temporal data (i.e., same mask for each time step)
        random_tensor = 1. - drop_prob
        if self.temporal_dropout:
            random_tensor = K.repeat_elements(
                random_tensor,
                self.input_spec.shape[1],
                1
            )

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


class MCDropout(Layer):
    """Applies Dropout to the input, even at prediction time.
    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.
    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from
        Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/
        srivastava14a.pdf)
        - [MCDropout: Dropout as a Bayesian Approximation: Representing Model
        Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142)
    """
    @interfaces.legacy_dropout_support
    def __init__(self,
                 dropout_rate=0.1,
                 temporal_dropout=False,
                 **kwargs):
        super(MCDropout, self).__init__(**kwargs)
        self.supports_masking = True
        self.temporal_dropout = temporal_dropout
        self.p = dropout_rate

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)

    def mc_dropout(self, inputs):
        '''Applies approx. dropping to inputs.'''
        # Parameters for concrete distribution
        eps = K.cast_to_floatx(K.epsilon())
        temp = 0.1

        # The dropout mask needs to be held constant, when Dropout is applied
        # to temporal data (i.e., same mask for each time step)
        if self.temporal_dropout:
            shape = (K.shape(inputs)[0], 1, K.shape(inputs)[2])
        else:
            shape = K.shape(inputs)

        # Use concrete distribution to calculate approx. dropout mask
        unif_noise = K.random_uniform(shape=shape)
        drop_prob = (
            K.log(self.p + eps) -
            K.cast(K.log(1. - self.p + eps), dtype='float32') +
            K.log(unif_noise + eps) -
            K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        retain_prob = 1. - self.p

        # The dropout mask needs to be held constant, when Dropout is applied
        # to temporal data (i.e., same mask for each time step)
        random_tensor = 1. - drop_prob
        if self.temporal_dropout:
            random_tensor = K.repeat_elements(
                random_tensor,
                self.input_spec.shape[1],
                1
            )

        # Approximately drop inputs
        inputs *= random_tensor
        inputs /= retain_prob

        return inputs

    def call(self, inputs, training=None):
        return self.mc_dropout(inputs)

    def get_config(self):
        config = {'regularizer': self.regularizer,
                  'init_range': self.init_range}
        base_config = super(MCDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Attention(Layer):
    """Applies simple static attention mechanism to tensor of shape
    (batch_size, time_steps, n_units). This can be applied for long-term
    prediction tasks by finding the the past state vectors of LSTMs/GRUs that
    are most relevant for the next prediction step.
    # References
        - [Modeling Long- and Short-Term Temporal Patterns with Deep Neural
        Networks](https://arxiv.org/abs/1703.07015)
    """

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

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
        super(Attention, self).build(input_shape)

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

    def __init__(self, channels, window, **kwargs):
        super(AutoRegression, self).__init__(**kwargs)
        self.hw = window
        self.channels = channels

    def build(self, input_shape):
        # Create a trainable kernel and bias for output.
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.hw, self.channels),
            initializer='uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            name='kernel',
            shape=(1, self.channels),
            initializer='uniform',
            trainable=True
        )
        super(AutoRegression, self).build(input_shape)

    def call(self, x):
        return K.sum(x[:, -self.hw:, :] * self.kernel, axis=1) + self.bias

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
