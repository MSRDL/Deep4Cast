# -*- coding: utf-8 -*-
"""Topologies module.

This module provides access to neural network topologies that can be used
insdide the forecaster module.

"""
import numpy as np
import keras.layers
from keras.models import Model

from . import custom_layers


class WaveNet(Model):
    """Extends keras.models.Model object.

    Implementation of WaveNet for multivariate time series.

    :param input_shape: Length and dimensionality of time series.
    :type input_shape: tuple
    :param output_shape: Output shape for predictions.
    :type output_shape: tuple
    :param topology: Neural network topology.
    :type topology: list
    :param dropout_rate:  Fraction of the units to drop for the linear
        transformation of the inputs. Float between 0 and 1.
    :type dropout_rate: float

    """
    def __init__(self, num_filters=32, num_layers=3, activation='relu'):
        """Initialize attributes."""
        if num_layers < 1:
            raise ValueError('num_layers must be > 1.')

        self.num_filters = num_filters
        self.num_layers = num_layers
        self.activation = activation

    def build_layers(self, input_shape, output_shape):
        """Build layers of the network.

        :param input_shape: Length and dimensionality of time series.
        :type input_shape: tuple
        :param output_shape: Output shape for predictions.
        :type output_shape: tuple

        """
        # First layer behaves differently cause of the difference in
        # channele for the conv laters.
        inputs, outputs = self.build_input(input_shape)

        # Core fof the network is created here
        for power in range(1, self.num_layers):
            outputs = self.build_wavenet_block(outputs, power)

        # Finally we need to match the output dimensions
        outputs = self.build_output(outputs, output_shape)

        super(WaveNet, self).__init__(inputs, outputs)

    def build_input(self, input_shape):
        """Return first layer of network."""
        inputs = keras.layers.Input(shape=input_shape)
        outputs = keras.layers.Conv1D(
            filters=self.num_filters,
            kernel_size=2,
            strides=1,
            padding='causal',
            dilation_rate=1,
            use_bias=True,
            name='dilated_1',
            activation=self.activation
        )(inputs)
        outputs = custom_layers.ConcreteDropout()(outputs)

        skip = keras.layers.SeparableConv1D(
            filters=self.num_filters,
            kernel_size=1,
            padding='same',
            name='skip_1',
            use_bias=True
        )(inputs)
        outputs = keras.layers.Add()([outputs, skip])

        return inputs, outputs

    def build_output(self, x, output_shape):
        """Return last layer for network."""
        outputs = keras.layers.SeparableConv1D(
            filters=output_shape,
            kernel_size=1,
            padding='same',
            name='skip_out',
            use_bias=True
        )(x)

        return outputs

    def build_wavenet_block(self, x, power):
        """Build core of the network."""
        outputs = keras.layers.Conv1D(
            filters=self.num_filters,
            kernel_size=2,
            strides=1,
            padding='causal',
            dilation_rate=2 ** power,
            use_bias=True,
            name='dilated_%d' % (2 ** power),
            activation=self.activation
        )(x)
        outputs = custom_layers.ConcreteDropout()(outputs)
        outputs = keras.layers.Add()([outputs, x])

        return outputs
