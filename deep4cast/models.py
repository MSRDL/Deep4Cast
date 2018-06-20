# -*- coding: utf-8 -*-
"""Neural network models module.

This module provides access to neural network models for different time
series forecasting purposes. These models function with uni- and
multi-variate time series alike.

"""
from inspect import getargspec

import keras.layers
import numpy as np
from keras.models import Model

from . import custom_layers


class SharedLayerModel(Model):
    """Extends keras.models.Model object.

    Include automatically constructed layers from input topology to
    make life easier when building regressors layer by layer and when
    optimizing network topology.

    :param input_shape: Length and dimensionality of time series.
    :type input_shape: tuple
    :param output_shape: Output shape for predictions.
    :type output_shape: tuple
    :param topology: Neural network topology.
    :type topology: list
    :param uncertainty: True if applying MC Dropout after every layer.
    :type uncertainty: boolean
    :param dropout_rate:  Fraction of the units to drop for the linear
        transformation of the inputs. Float between 0 and 1.
    :type dropout_rate: float

    """

    def __init__(self,
                 input_shape,
                 output_shape,
                 topology,
                 uncertainty=False,
                 concreteDropout=False,
                 dropout_rate=0.1):
        """Initialize attributes."""
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._topology = topology
        self._rnd_init = 'glorot_normal'
        self._uncertainty = uncertainty
        self._concreteDropout = concreteDropout
        self._dropout_rate = dropout_rate

        # Initialize super class with custom layers
        inputs, outputs = self._build_layers()
        super(SharedLayerModel, self).__init__(inputs, outputs)

    def _build_layers(self):
        """Build model layers one layer at a time from topology."""
        layers = {}
        rate = self._dropout_rate

        # Create an input layer using shape of data
        layers['input'] = keras.layers.Input(shape=self._input_shape)

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

                if self._uncertainty:
                    # If MC Dropout, aka the Bayesian approximation to Neural
                    # Networks should be used, we add Dropout after each layer,
                    # even at test time.
                    dropout_id = next_id + '_' + layer_id + '_dropout'
                    if self._concreteDropout:
                        layers[dropout_id] = custom_layers.ConcreteDropout(
                            layers[next_id]
                        )
                    else:
                        layers[dropout_id] = custom_layers.MCDropout(rate)(
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
        if self._uncertainty:
            dropout_id = layer_id + '_dense_dropout'
            if self._concreteDropout:
                layers[dropout_id] = custom_layers.ConcreteDropout(
                    layers[layer_id]
                )
            else:
                layers[dropout_id] = custom_layers.MCDropout(rate)(
                    layers[layer_id]
                )
            layer_id = dropout_id

        layer_cls = keras.layers.Dense
        params = {'units': np.prod(self._output_shape)}
        layers['_dense'] = layer_cls(**params)(layers[layer_id])

        layer_cls = keras.layers.Reshape
        params = {'target_shape': self._output_shape}
        layers['output'] = layer_cls(**params)(layers['_dense'])

        return layers['input'], layers['output']
