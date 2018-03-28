# -*- coding: utf-8 -*-
"""Temporal convolutional neural network module.

This module provides access to temporal CNN models for different time
series forecasting purposes. These models function with uni- and
multi-variate time series alike.

"""
from copy import copy
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.models import Sequential


class TemporalCNN(Sequential):
    """Extends keras.models.Sequential object.

    Include automatically constructed CNN layers from input topology to
    make life easier when building regressors and for optimizing network
    topology.

    :param input_shape: Length and dimesnionality of time series
    :type input_shape: tuple of ints
    :param topology: Neural network topology.
    :type topology: list of integer tuples of length 4
    :raises: Exception in case of misspecified topolgy.

    """

    def __init__(self, input_shape, topology):
        """Initialize additional properties."""
        self._input_shape = input_shape
        self._topology = topology
        self._rnd_init = 'glorot_normal'
        self._activation = 'elu'

        # Check if topology conforms to exepcted model topology kind
        self.check_topology(topology)

        # Initialize super class with custom layers
        super(TemporalCNN, self).__init__(self._build_layers())

    def _build_layers(self):
        """Build CNN model layers.

        Add CNN layers one at a time with layerwise output for easy
        inspection of internal layers after model is trained.

        """
        n_layers = len(self._topology)
        layers = []
        for i in range(n_layers):
            n_out, kernel_size, pool_size, pool_stride = self._topology[i]
            layer_conv = Conv1D(
                n_out,
                kernel_size,
                activation=self._activation,
                kernel_initializer=self._rnd_init,
                input_shape=self._input_shape
            )
            layer_max = MaxPooling1D(
                pool_size,
                pool_stride
            )
            layers.append(layer_conv)
            layers.append(layer_max)

        # Add dense and dropout layers
        layers.append(Flatten())
        layers.append(Dense(128, kernel_initializer=self._rnd_init))
        layers.append(Dense(self._input_shape[1]))

        return layers

    @staticmethod
    def check_topology(topology):
        """Raise error if topology has wrong format for type of network.

        :param topology: Neural network topology.
        :type topology: list of integer tuples of length 4
        :raises: Exception in case of misspecified topology.

        """
        # Check if topology is not empty
        if not topology:
            raise Exception('Topology list is empty.')

        # Check if list items are tuples
        if any(not isinstance(layer, tuple) for layer in topology):
            raise Exception(
                'Topology should be list of postive integers tuples.'
            )

        # Check if each list item tuple contains 3 elements
        if any(len(layer) != 4 for layer in topology):
            raise Exception(
                'Each layer tuple should contain 4 positive integers.'
            )

        # Check if each list item tuple contains only integers
        if any(any(not isinstance(n, int) for n in layer) for layer in topology):
            raise Exception(
                'Topology should be list of postive integers tuples.'
            )
        # Check if each list item tuple contains only positive elements
        if any(any(n < 1 for n in layer) for layer in topology):
            raise Exception(
                'Topology should be list of postive integers tuples.'
            )
