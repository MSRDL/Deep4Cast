# -*- coding: utf-8 -*-
"""Topologies module.

This module provides access to neural network topologies that can be used
insdide the forecaster module.

"""

_CNN = [
    {
        'meta': {
            'layer_type': 'Conv1D',
            'layer_id': 'conv1',
            'parent_ids': ['input']
        },
        'params': {
            'filters': 256,
            'kernel_size': 6,
            'activation': 'elu'
        }
    },
    {
        'meta': {
            'layer_type': 'Conv1D',
            'layer_id': 'conv2',
            'parent_ids': ['conv1']
        },
        'params': {
            'filters': 128,
            'kernel_size': 5,
            'activation': 'elu'
        }
    },
    {
        'meta': {
            'layer_type': 'Conv1D',
            'layer_id': 'conv3',
            'parent_ids': ['conv2']
        },
        'params': {
            'filters': 64,
            'kernel_size': 4,
            'activation': 'elu'
        }
    },
    {
        'meta': {
            'layer_type': 'Flatten',
            'layer_id': 'flat1',
            'parent_ids': ['conv3']
        },
        'params': {}
    },
    {
        'meta': {
            'layer_type': 'Dense',
            'layer_id': 'dense1',
            'parent_ids': ['flat1']
        },
        'params': {
            'units': 128,
            'activation': 'elu'
        }
    },
    {
        'meta': {
            'layer_type': 'Dense',
            'layer_id': 'dense2',
            'parent_ids': ['dense1']
        },
        'params': {
            'units': 256,
            'activation': 'elu'
        }
    }
]

_GRU = [
    {
        'meta': {
            'layer_type': 'GRU',
            'layer_id': 'gru1',
            'parent_ids': ['input']
        },
        'params': {
            'units': 128,
            'return_sequences': False
        }
    }
]

_LSTM = [
    {
        'meta': {
            'layer_type': 'LSTM',
            'layer_id': 'lstm1',
            'parent_ids': ['input']
        },
        'params': {
            'units': 128,
            'return_sequences': False
        }
    }
]

_LSTNET = [
    {
        'meta': {
            'layer_type': 'Conv1D',
            'layer_id': 'conv1',
            'parent_ids': ['input']
        },
        'params': {
            'filters': 128, 
            'kernel_size': 5
        }
    },    
    {
        'meta': {
            'layer_type': 'GRU',
            'layer_id': 'gru1',
            'parent_ids': ['conv1']
        },
        'params': {
            'units': 128, 
            'return_sequences': True
        }
    },
    {
        'meta': {
            'layer_type': 'TemporalAttention',
            'layer_id': 'att1',
            'parent_ids': ['gru1']
        },
        'params': {}
    },
    {
        'meta': {
            'layer_type': 'AutoRegression',
            'layer_id': 'ar1',
            'parent_ids': ['input']
        },
        'params': {}
    },
    {
        'meta': {
            'layer_type': 'Concatenate',
            'layer_id': 'con1',
            'parent_ids': ['ar1', 'att1']
        },
        'params': {}
    }
]


def get_topology(type):
    """General forecaster class for forecasting time series.
    :param type: Neural network topolgy kind.
    :type type: list

    """
    if type.lower() == 'cnn':
        return _CNN
    elif type.lower() == 'gru':
        return _GRU
    elif type.lower() == 'lstm':
        return _LSTM
    elif type.lower() == 'lstnet':
        return _LSTNET
    else:
        raise ValueError('Unknown topology kind.')
