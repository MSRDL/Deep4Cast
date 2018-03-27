# -*- coding: utf-8 -*-
"""
Multivariate recurrent neural network module.

This module provides access to multivariate RNN models for different time
series forecasting purposes. These models function with uni- and
multi-variate time series alike.

"""
from keras.layers import Dense, GRU
from keras.models import Sequential


class MultivariateRNN(Sequential):
    """
    Extend Sequential model object.

    Include automatically constructed RNN layers from input topology to
    make life easier when building regressors and embedders.

    :param input_dim: Dimension of multivariate time series.
    :type input_dim: int
    :param input_length: Length of input time series.
    :type input_length: float
    :param topology: Neural network topology.
    :type topology: dict
    :param unit: Recurrent neural network cell.
    :type unit: keras.layers.recurrent.RNN
    :raises: ValueError in case of misspecified topolgy.

    """

    def __init__(self, input_dim, input_length, topology, unit=GRU):
        """Initialize additional properties."""
        self._input_dim = input_dim
        self._input_length = input_length
        self._topology = topology
        self._unit = unit
        self._rnd_init = 'glorot_normal'

        # Check if topology conforms to exepcted topology type
        self._check_topology(self._topology)

        # Initialize super class with custom layers
        super(MultivariateRNN, self).__init__(self._build_layers)

    def _build_layers(self):
        """
        Build RNN model layers.

        Add RNN layers one at a time with layerwise output for easy
        inspection of internal layers after model is trained.
        """
        n_layers = len(self._topology)
        layers = []
        for i in range(n_layers):
            return_sequences = True if i < n_layers - 1 else False
            layer = self.__unit(
                self._topology[i],
                input_length=self._input_length,
                input_dim=self._input_dim,
                return_sequences=return_sequences
            )
            layers.append(layer)

        # Add dense output later
        layers.append(Dense(self._input_dim, init=self._rnd_init))

        return layers

    @property
    def unit(self):
        """Return RNN base unit."""
        return self._unit.__name__

    @unit.setter
    def unit(self, unit):
        """Set the unit for the network architecture safely."""
        self._unit = unit

        # Reinitialize the network architecture
        print('Model reinitialized.')
        super(MultivariateRNN, self).__init__(self._build_layers)

    @property
    def topology(self):
        """Return RNN topology."""
        return self._topology

    @topology.setter
    def topology(self, topology):
        """Set the unit for the network architecture safely."""
        self._topology = topology

        # Check if topology conforms to exepcted topology type
        self._check_topology(self._topology)

        # Reinitialize the network architecture
        print('Model reinitialized.')
        super(MultivariateRNN, self).__init__(self._build_layers)

    @staticmethod
    def _check_topology(topology):
        """Check if topology has correct format for type of network."""
        if list(topology.keys()) != ['n_layers', 'n_units']:
            raise ValueError('Topology should be n_layers, n_units dict.')

        if type(topology['n_layers']) != int:
            raise ValueError('Topology should contain int n_layers.')

        if topology['n_layers'] < 1:
            raise ValueError('Topology should have at least 1 layer.')

        if type(topology['n_units']) != list:
            raise ValueError('Topology should contain list n_units.')

        if len(topology['n_units']) < 1:
            raise ValueError('Topology should have at least 1 unit per layer')
