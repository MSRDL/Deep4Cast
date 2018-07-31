# -*- coding: utf-8 -*-
"""Topologies module.

This module provides access to neural network topologies that can be used
insdide the forecaster module.

"""
from inspect import getargspec

import numpy as np
import keras.layers
from keras.models import Model

from . import custom_layers


class LSTNet(Model):
    """Extends keras.models.Model object.

    Implementation of LSTNet topology for multivariate time series. 
    See https://arxiv.org/pdf/1703.07015.pdf

    :param conv_filters: Number of filters for conv layer.
    :type conv_filters: int
    :param rnn_units: Number of hidden RNN units.
    :type rnn_units: int
    :param activation: Activation function.
    :type activation: string

    """
    def __init__(self, conv_filters=32, rnn_units=32, activation='relu'):
        """Initialize attributes."""
        self.conv_filters = conv_filters
        self.rnn_units = rnn_units
        self.activation = activation
        self._dropout_layer = custom_layers.ConcreteDropout

    def build_layers(self, input_shape, output_shape):
        """Build layers of the network.

        :param input_shape: Length and dimensionality of time series.
        :type input_shape: tuple
        :param output_shape: Output shape for predictions.
        :type output_shape: tuple

        """
        # First layer behaves differently cause of the difference in
        # channels for the conv laters.
        inputs = keras.layers.Input(shape=input_shape)

        # Convolutional layer
        outputs = keras.layers.Conv1D(
            filters=self.conv_filters,
            kernel_size=2,
            strides=1,
            padding='causal',
            dilation_rate=1,
            use_bias=True,
            activation=self.activation
        )(inputs)
        outputs = self._dropout_layer()(outputs)

        # Recurrent component
        outputs = keras.layers.GRU(
            units=self.rnn_units,
            activation=self.activation,
            return_sequences=True
        )(outputs)
        outputs = self._dropout_layer()(outputs)

        # Attention layer
        outputs = custom_layers.Attention()(outputs)
        outputs = self._dropout_layer()(outputs)

        # Autoregressive component
        ar = custom_layers.AutoRegression()(inputs)
        ar = self._dropout_layer()(ar)

        # Finalize output
        outputs = keras.layers.Add()([outputs, ar])
        outputs = self._dropout_layer()(outputs)
        outputs = keras.layers.Dense(units=np.prod(output_shape))(outputs)
        outputs = keras.layers.Reshape(target_shape=output_shape)(outputs)

        super(LSTNet, self).__init__(inputs, outputs)


class StackedGRU(Model):
    """Extends keras.models.Model object.

    Implementation of stacked GRU topology for multivariate time series.

    :param units: Number of hidden units for each layer.
    :type units: int
    :param num_layers: Number of stacked layers.
    :type num_layers: int
    :param activation: Activation function.
    :type activation: string

    """
    def __init__(self, units=32, num_layers=1, activation='relu'):
        """Initialize attributes."""
        if num_layers < 1:
            raise ValueError('num_layers must be > 1.')

        self.units = units
        self.num_layers = num_layers
        self.activation = activation
        self._dropout_layer = custom_layers.ConcreteDropout

    def build_layers(self, input_shape, output_shape):
        """Build layers of the network.

        :param input_shape: Length and dimensionality of time series.
        :type input_shape: tuple
        :param output_shape: Output shape for predictions.
        :type output_shape: tuple

        """
        inputs, outputs = self.build_input(input_shape)

        # Core of the network is created here
        for power in range(1, self.num_layers):
            outputs = self.build_gru_block(outputs)

        # Finally we need to match the output dimensions
        outputs = self.build_output(outputs, output_shape)

        super(StackedGRU, self).__init__(inputs, outputs)

    def build_input(self, input_shape):
        """Return first layer of network."""
        inputs = keras.layers.Input(shape=input_shape)
        outputs = keras.layers.GRU(
            units=self.units,
            activation=self.activation
        )(inputs)
        outputs = self._dropout_layer()(outputs)

        skip = keras.layers.SeparableConv1D(
            filters=self.units,
            kernel_size=1,
            padding='same',
            name='skip_1',
            use_bias=True
        )(inputs)
        outputs = keras.layers.Add()([outputs, skip])

        return inputs, outputs

    def build_output(self, x, output_shape):
        """Return last layer for network."""
        outputs = keras.layers.Conv1D(
            filters=output_shape[1],
            kernel_size=1,
            padding='same',
            name='skip_out',
            use_bias=True
        )(x)
        outputs = keras.layers.Flatten()(outputs)
        outputs = self._dropout_layer()(outputs)
        outputs = keras.layers.Dense(units=np.prod(output_shape))(outputs)
        outputs = keras.layers.Reshape(target_shape=output_shape)(outputs)

        return outputs

    def build_gru_block(self, x):
        """Build core of the network."""
        outputs = keras.layers.GRU(
            units=self.units,
            activation=self.activation
        )(x)
        outputs = self._dropout_layer()(outputs)
        outputs = keras.layers.Add()([outputs, x])

        return outputs


class WaveNet(Model):
    """Extends keras.models.Model object.

    Implementation of WaveNet-like topology for multivariate time series.

    :param filters: Number of hidden units for each layer.
    :type filters: int
    :param num_layers: Number of stacked layers.
    :type num_layers: int
    :param activation: Activation function.
    :type activation: string

    """

    def __init__(self, filters=32, num_layers=1, activation='relu'):
        """Initialize attributes."""
        if num_layers < 1:
            raise ValueError('num_layers must be > 1.')

        self.filters = filters
        self.num_layers = num_layers
        self.activation = activation
        self._dropout_layer = custom_layers.ConcreteDropout

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
            filters=self.filters,
            kernel_size=2,
            strides=1,
            padding='causal',
            dilation_rate=1,
            use_bias=True,
            name='dilated_1',
            activation=self.activation
        )(inputs)
        outputs = self._dropout_layer()(outputs)

        skip = keras.layers.SeparableConv1D(
            filters=self.filters,
            kernel_size=1,
            padding='same',
            name='skip_1',
            use_bias=True
        )(inputs)
        outputs = keras.layers.Add()([outputs, skip])

        return inputs, outputs

    def build_output(self, x, output_shape):
        """Return last layer for network."""
        outputs = keras.layers.Conv1D(
            filters=output_shape[1],
            kernel_size=1,
            padding='same',
            name='skip_out',
            use_bias=True
        )(x)
        outputs = keras.layers.Flatten()(outputs)
        outputs = self._dropout_layer()(outputs)
        outputs = keras.layers.Dense(units=np.prod(output_shape))(outputs)
        outputs = keras.layers.Reshape(target_shape=output_shape)(outputs)

        return outputs

    def build_wavenet_block(self, x, power):
        """Build core of the network."""
        outputs = keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=2,
            strides=1,
            padding='causal',
            dilation_rate=2 ** power,
            use_bias=True,
            name='dilated_%d' % (2 ** power),
            activation=self.activation
        )(x)
        outputs = self._dropout_layer()(outputs)
        outputs = keras.layers.Add()([outputs, x])

        return outputs


class CustomTopologyModel(Model):
    """Extends keras.models.Model object.

    Include automatically constructed layers from input topology to
    make life easier when building regressors layer by layer and when
    optimizing network topology.

    :param topology: Neural network topology.
    :type topology: list

    """

    def __init__(self, topology):
        """Initialize attributes."""
        self._topology = topology
        self._rnd_init = 'glorot_normal'
        self._dropout_layer = custom_layers.ConcreteDropout

    def build_layers(self, input_shape, output_shape):
        """Build model layers one layer at a time from topology."""
        layers = {}

        # Create an input layer using shape of data
        layers['input'] = keras.layers.Input(shape=input_shape)

        # Loop over all layers and construct layer relationships using
        # Keras functional API for complex networks.
        for layer in self._topology:
            # Unpack relevant items in params and meta to reduce visual noise
            layer_type = layer['meta']['layer_type']
            layer_id = layer['meta']['layer_id']
            parent_ids = layer['meta']['parent_ids']
            params = layer['params']

            # Construct layer to be built from string 'meta' which is
            # part of topology list.
            try:
                layer_cls = getattr(keras.layers, layer_type)
            except AttributeError:
                layer_cls = getattr(custom_layers, layer_type)

            # In order to get the accepteble arguments for each layer we need
            # to use inspect because of keras' legacy support decorators.
            try:
                args = getargspec(layer_cls.__init__._original_function)[0]
            except AttributeError:
                args = getargspec(layer_cls)[0]

            # Add other input arguments to params dict when
            # necessary. For example, MaxPooling1D does not accept the
            # argument 'kernel_initializer'.
            if 'kernel_initializer' in args:
                params['kernel_initializer'] = self._rnd_init

            # Create the Keras layer using a switch for MC Dropout after
            # every layer.
            parents = []
            for parent_id in parent_ids:
                next_id = parent_id

                # If MC Dropout, aka the Bayesian approximation to Neural
                # Networks should be used, we add Dropout after each layer,
                # even at test time.
                dropout_id = next_id + '_' + layer_id + '_dropout'
                layers[dropout_id] = self._dropout_layer()(
                    layers[next_id]
                )
                next_id = dropout_id

                parents.append(layers[next_id])

            # Many layers don't expect a list of tensors as input but just
            # a single tensor, but the Concat layer expects a list of input
            # tensors, so we need to deal with this case.
            if len(parents) < 2:
                parents = parents[0]

            layers[layer_id] = layer_cls(**params)(parents)

        # Need to handle the output layer and reshaping for multi-step
        # forecasting.
        dropout_id = layer_id + '_dense_dropout'
        layers[dropout_id] = self._dropout_layer()(
            layers[layer_id]
        )
        layer_id = dropout_id

        layer_cls = keras.layers.Dense
        params = {
            'units': np.prod(output_shape),
            'kernel_initializer': 'glorot_normal'
        }
        layers['_dense'] = layer_cls(**params)(layers[layer_id])

        layer_cls = keras.layers.Reshape
        params = {'target_shape': output_shape}
        layers['output'] = layer_cls(**params)(layers['_dense'])

        # Initialize super class with custom layers
        super(CustomTopologyModel, self).__init__(
            layers['input'],
            layers['output']
        )
