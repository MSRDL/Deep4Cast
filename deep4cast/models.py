# -*- coding: utf-8 -*-
"""Neural network models module.

This module provides access to neural network models for different time
series forecasting purposes. These models function with uni- and
multi-variate time series alike.

"""
import keras.layers
from inspect import getargspec
from keras import backend as K
from keras.layers.core import Lambda
from keras.models import Sequential


class LayeredTimeSeriesModel(Sequential):
    """Extends keras.models.Sequential object.

    Include automatically constructed layers from input topology to
    make life easier when building regressors layer by layer and when
    optimizing network
    topology.

    :param input_shape: Length and dimensionality of time series.
    :type input_shape: tuple
    :param topology: Neural network topology
    :type topology: list

    """

    def __init__(self, input_shape, topology=None, dropout_rate=0.5):
        """Initialize attributes."""
        self._input_shape = input_shape
        self._topology = topology
        self._rnd_init = 'glorot_normal'
        self._dropout_rate = dropout_rate

        # Initialize super class with custom layers
        super(LayeredTimeSeriesModel, self).__init__(self._build_layers())

    def _build_layers(self):
        """Build model layers one layer at a time from topology."""
        layers = []
        if self._topology:
            n_layers = len(self._topology)
            for i, (name, params) in enumerate(self._topology):
                # Construct Keras layer to be built from string 'name' which is
                # part of topology list.
                layer_cls = getattr(keras.layers, name)

                # In order to get the accepteble arguments for each layer we need
                # to use inspect because of keras's legacy support decorators.
                try:
                    args = getargspec(layer_cls.__init__._original_function)[0]
                except AttributeError:
                    args = getargspec(layer_cls)[0]

                # Add activation and other input arguments to params dict when
                # necessary. For example, MaxPooling1D does not
                # accept the argument 'kernel_initializer'.
                if 'kernel_initializer' in args:
                    params['kernel_initializer'] = self._rnd_init

                # The 'return_squences' argument is necessary for stacking RNNs.
                if 'return_sequences' in args and i < n_layers - 1:
                    params['return_sequences'] = True

                # Create Keras layer, but make sure that the input shape is
                # passed to the first later.
                if i == 0:
                    layer = layer_cls(**params, input_shape=self._input_shape)
                else:
                    layer = layer_cls(**params)
                layers.append(layer)

        # Add at least a dense layer for output with output shape of the
        # time series dimensionality.
        params = {}
        params['units'] = self._input_shape[1]
        if not self._topology:
            params['input_shape'] = self._input_shape
        else:
            layers.append(Lambda(lambda x: K.dropout(x, level=self._dropout_rate)))
        layers.append(keras.layers.Dense(**params))

        return layers
