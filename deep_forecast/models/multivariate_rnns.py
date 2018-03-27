# -*- coding: utf-8 -*-
"""Multivariate recurrent neural network module.

This module provides access to multivariate RNN models for different time
series forecasting purposes. These models function with uni- and
multi-variate time series alike.

"""
from keras.layers import Dense, GRU
from keras.models import Sequential


class TruncatedRNN(Sequential):
    """Extends Sequential model object.

    Include automatically constructed RNN layers from input topology to
    make life easier when building regressors and for optimizing network
    topology.

    :param input_dim: Dimension of multivariate time series.
    :type input_dim: int
    :param input_length: Length of input time series.
    :type input_length: float
    :param topology: Neural network topology.
    :type topology: list of integers for RNN units per layer
    :param unit: Recurrent neural network cell.
    :type unit: keras.layers.recurrent.RNN
    :raises: Exception in case of misspecified topolgy.

    """

    def __init__(self, input_dim, input_length, topology, rnn_unit=GRU):
        """Initialize additional properties."""
        self._input_dim = input_dim
        self._input_length = input_length
        self._topology = topology
        self._rnn_unit = rnn_unit
        self._rnd_init = 'glorot_normal'

        # Check if topology conforms to exepcted model topology kind
        self.check_topology(topology)

        # Initialize super class with custom layers
        super(TruncatedRNN, self).__init__(self._build_layers)

    def _build_layers(self):
        """Build RNN model layers.

        Add RNN layers one at a time with layerwise output for easy
        inspection of internal layers after model is trained.

        """
        n_layers = len(self._topology)
        layers = []
        for i in range(n_layers):
            return_sequences = True if i < n_layers - 1 else False
            layer = self._rnn_unit(
                self._topology[i],
                input_length=self._input_length,
                input_dim=self._input_dim,
                return_sequences=return_sequences
            )
            layers.append(layer)

        # Add dense output layer
        layers.append(Dense(self._input_dim, init=self._rnd_init))

        return layers

    @staticmethod
    def check_topology(topology):
        """Raise error if topology has wrong format for type of network.

        :param topology: Neural network topology.
        :type topology: list of integers for RNN units per layer.
        :raises: ValueError in case of misspecified topolgy.

        """
        if not topology:
            raise Exception('Topology list is empty.')

        if any(not isinstance(n, int) or n < 1 for n in topology):
            raise Exception('Topology should be list of positive integers.')
