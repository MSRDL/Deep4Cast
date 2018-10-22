# -*- coding: utf-8 -*-
"""Custom layers module."""
import numpy as np

from keras import initializers
from keras import backend as K
from keras.layers import Layer
from keras.legacy import interfaces
from keras.engine import InputSpec


class ConcreteDropout(Layer):
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
    @interfaces.legacy_dropout_support
    def __init__(self,
                 temporal=False,
                 reg=1e-5,
                 init_range=(0.1, 0.3),
                 **kwargs):
        """Initialize parameters."""
        super(ConcreteDropout, self).__init__(**kwargs)
        self.reg = reg
        self.init_range = init_range
        self.supports_masking = True
        self.temporal = temporal

        # Dropout regularization parameters
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

        # The input dim used to set up the dropout probability reg,
        # depends in whether the dropout mask is constant accross time or not
        if self.temporal:
            input_dim = self.input_spec.shape[-1]  # Drop only last dim
        else:
            input_dim = np.prod(input_shape[1:])  # Drop only last two dims

        reg = self.p * K.log(self.p)
        reg += (1. - self.p) * K.log(1. - self.p)
        reg *= self.reg * input_dim
        self.add_loss(reg)

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
        if self.temporal:
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
        if self.temporal:
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
        config = {'reg': self.reg,
                  'init_range': self.init_range}
        base_config = super(ConcreteDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class MCDropout(Layer):
    """Applies Dropout to the input, even at prediction time.

    Dropout consists in randomly setting a fraction `rate` of input units to 0 
    at each update during training time, which helps prevent overfitting. At 
    prediction time the units are then also dropped out with the same fraction.
    This generates samples from an approximate posterior predictive 
    distribution.

    References
        - [Dropout: A Simple Way to Prevent Neural Networks from
        Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/
        srivastava14a.pdf)
        - [MCDropout: Dropout as a Bayesian Approximation: Representing Model
        Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142)

    """
    @interfaces.legacy_dropout_support
    def __init__(self, rate=0.1, temporal=False, **kwargs):
        """Initialize parameters."""
        super(MCDropout, self).__init__(**kwargs)
        self.supports_masking = True
        self.temporal = temporal  # Apply the same dropout mask across all time
        self.p = rate

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)

    def mc_dropout(self, inputs):
        '''Applies approx. dropping to inputs.'''
        # Parameters for concrete distribution
        eps = K.cast_to_floatx(K.epsilon())
        temp = 0.1

        # The dropout mask needs to be held constant, when Dropout is applied
        # to temporal data (i.e., same mask for each time step)
        if self.temporal:
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
        # to temporal data , i.e., same mask for each time step
        random_tensor = 1. - drop_prob
        if self.temporal:
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

    def compute_output_shape(self, input_shape):
        return input_shape
