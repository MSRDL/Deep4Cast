# -*- coding: utf-8 -*-
"""Topologies module.

This module provides access to neural network topologies that can be used
insdide the forecaster module.

"""

_CNN = [
    {
        'meta': {
            'layer_type': 'Conv1D',
            'layer_id': 'c1',
            'parent_ids': ['input']
        },
        'params': {
            'filters': 64,
            'kernel_size': 5,
            'activation': 'elu'
        }
    },
    {
        'meta': {
            'layer_type': 'Conv1D',
            'layer_id': 'c2',
            'parent_ids': ['c1']
        },
        'params': {
            'filters': 64,
            'kernel_size': 3,
            'activation': 'elu'
        }
    },
    {
        'meta': {
            'layer_type': 'Conv1D',
            'layer_id': 'c3',
            'parent_ids': ['c2']
        },
        'params': {
            'filters': 128,
            'kernel_size': 3,
            'activation': 'elu'
        }
    },
    {
        'meta': {
            'layer_type': 'Flatten',
            'layer_id': 'f1',
            'parent_ids': ['c3']
        },
        'params': {}
    },
    {
        'meta': {
            'layer_type': 'Dense',
            'layer_id': 'd1',
            'parent_ids': ['f1']
        },
        'params': {
            'units': 128,
            'activation': 'elu'
        }
    },
    {
        'meta': {
            'layer_type': 'Dense',
            'layer_id': 'd2',
            'parent_ids': ['d1']
        },
        'params': {
            'units': 128,
            'activation': 'elu'
        }
    }
]


_RNN = [
    {
        'meta': {
            'layer_type': 'Conv1D',
            'layer_id': 'c1',
            'parent_ids': ['input']
        },
        'params': {
            'filters': 64,
            'kernel_size': 5,
            'activation': 'elu'
        }
    },
    {
        'meta': {
            'layer_type': 'Conv1D',
            'layer_id': 'c2',
            'parent_ids': ['c1']
        },
        'params': {
            'filters': 64,
            'kernel_size': 3,
            'activation': 'elu'
        }
    },
    {
        'meta': {
            'layer_type': 'Conv1D',
            'layer_id': 'c3',
            'parent_ids': ['c2']
        },
        'params': {
            'filters': 128,
            'kernel_size': 3,
            'activation': 'elu'
        }
    },
    {
        'meta': {
            'layer_type': 'Flatten',
            'layer_id': 'f1',
            'parent_ids': ['c3']
        },
        'params': {}
    },
    {
        'meta': {
            'layer_type': 'Dense',
            'layer_id': 'd1',
            'parent_ids': ['f1']
        },
        'params': {
            'units': 128,
            'activation': 'elu'
        }
    },
    {
        'meta': {
            'layer_type': 'Dense',
            'layer_id': 'd2',
            'parent_ids': ['d1']
        },
        'params': {
            'units': 128,
            'activation': 'elu'
        }
    }
]


_RNN = [
    {
        'meta': {
            'layer_type': 'GRU',
            'layer_id': 'g1',
            'parent_ids': ['input']
        },
        'params': {
            'units': 128,
            'return_sequences': True
        }
    },
    {
        'meta': {
            'layer_type': 'GRU',
            'layer_id': 'g2',
            'parent_ids': ['g1']
        },
        'params': {
            'units': 128,
            'return_sequences': False
        }
    }
]


def get_topology(type):
    """General forecaster class for forecasting time series.
    :param type: Neural network topolgy kidn.
    :type type: list

    """
    if type.lower() == 'cnn':
        return _CNN
    elif type.lower() == 'rnn':
        return _RNN
    else:
        raise ValueError('Unknown topology kind.')
