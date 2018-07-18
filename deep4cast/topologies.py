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

    def __init__(self, num_filters=32, num_layers=3, activation='relu'):
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.activation = activation

    def build(self, input_shape, output_shape, targets=None):
        inputs, outputs = self.build_input(input_shape, targets)
        for power in self.num_layers:
            outputs = self.build_wavenet_block(outputs, power)
        outputs = self.build_output(outputs, output_shape, targets)
        
        super(WaveNet, self).__init__(inputs, outputs)

    def build_input(self, input_shape, targets=None):
        print(input_shape)
        inputs = keras.layers.Input(shape=self.input_shape)
        outputs = inputs
        dilated_conv = keras.layers.Conv1D(
            filters=self.num_filters,
            kernel_size=2,
            strides=1,
            padding='causal',
            dilation_rate=1,
            bias=True,
            name='Dilated_Conv1D_0',
            activation=self.activation
        )(outputs)

        skip = keras.layers.Conv1D(
            filters=self.num_filters,
            kernel_size=1,
            border_mode='same',
            bias=True
        )(outputs)
        outputs = keras.layers.Merge(mode='sum')([dilated_conv, skip])
        outputs = custom_layers.ConcreteDropout(outputs)

        return inputs, outputs

    def build_output(self, x, output_shape):
        outputs = keras.layers.Conv1D(
            filters=self.num_filters,
            kernel_size=1,
            border_mode='same',
            bias=True
        )(x)
        outputs = custom_layers.ConcreteDropout(outputs)
        outputs = keras.layers.Dense(units=np.prod(output_shape))(outputs)
        outputs = keras.layers.Reshape(target_shape=output_shape)(outputs)

        return outputs

    def build_wavenet_block(self, x, power):
        x_in = x
        dilated_conv = keras.layers.Conv1D(
            filters=self.num_filters,
            kernel_size=2,
            strides=1,
            padding='causal',
            dilation_rate=2 ** power,
            bias=True,
            name='Dilated_Conv1D_%d' % (2 ** power),
            activation=self.activation
        )(x)
        outputs = keras.layers.Merge(mode='sum')([dilated_conv, x_in])
        outputs = custom_layers.ConcreteDropout(outputs)

        return outputs

    def compute_receptive_field_(dilation_depth, nb_stacks):
        receptive_field = nb_stacks * (2 ** dilation_depth * 2) - (nb_stacks - 1)
        receptive_field_ms = (receptive_field * 1000) / desired_sample_rate
    return receptive_field, receptive_field_ms