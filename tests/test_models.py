# -*- coding: utf-8 -*-

from fixtures import synthetic_data

from deep4cast.models import SharedLayerModel


def test_defaults(synthetic_data):
    # Get the data
    X = synthetic_data

    topology = [
        {
            'meta': {
                'layer_type': 'GRU',
                'layer_id': 'gru1',
                'parent_ids': ['input']},
            'params': {
                'units': 128,
                'return_sequences': True
            }
        },
        {
            'meta': {
                'layer_type': 'GRU',
                'layer_id': 'gru2',
                'parent_ids': ['gru1']
            },
            'params': {
                'units': 128,
                'return_sequences': False
            }
        }
    ]

    # Instantiate model
    input_shape = X.shape[:2]
    output_shape = (21, 2)
    model = SharedLayerModel(input_shape, output_shape, topology)

    # Check all (!) instance parameters
    assert model._topology == topology
    assert model._input_shape == X.shape[:2]
    assert model._output_shape == (21, 2)
    assert model._rnd_init == 'glorot_normal'
    assert not model._uncertainty
    assert model._dropout_rate == 0.1
    assert len(model.layers) == len(topology) + 3  # Account for input layer


def test_defaults_mcdropout(synthetic_data):
    # Get the data
    X = synthetic_data

    topology = [
        {
            'meta': {
                'layer_type': 'Conv1D',
                'layer_id': 'c1',
                'parent_ids': ['input']},
            'params': {
                'filters': 64,
                'kernel_size': 3
            }
        },
        {
            'meta': {
                'layer_type': 'GRU',
                'layer_id': 'gru2',
                'parent_ids': ['c1']
            },
            'params': {
                'units': 128,
                'return_sequences': False
            }
        }
    ]

    # Instantiate model
    input_shape = X.shape[:2]
    output_shape = (21, 3)
    model = SharedLayerModel(
        input_shape,
        output_shape,
        topology,
        uncertainty=True
    )

    # Check all (!) instance parameters
    assert model._topology == topology
    assert model._input_shape == X.shape[:2]
    assert model._output_shape == (21, 3)
    assert model._rnd_init == 'glorot_normal'
    assert model._uncertainty
    assert model._dropout_rate == 0.1
    assert len(model.layers) == len(topology) + 6  # Account for input layer


def test_concat_default(synthetic_data):
    # Get the data
    X = synthetic_data

    topology = [
        {
            'meta': {
                'layer_type': 'GRU',
                'layer_id': 'gru1',
                'parent_ids': ['input']},
            'params': {
                'units': 128,
                'return_sequences': False
            }
        },
        {
            'meta': {
                'layer_type': 'GRU',
                'layer_id': 'gru2',
                'parent_ids': ['input']
            },
            'params': {
                'units': 128,
                'return_sequences': False
            }
        },
        {
            'meta': {
                'layer_type': 'Concatenate',
                'layer_id': 'concat1',
                'parent_ids': ['gru1', 'gru2']
            },
            'params': {
                'axis': -1
            }
        }
    ]

    # Instantiate model
    input_shape = X.shape[:2]
    output_shape = (21, 2)
    model = SharedLayerModel(
        input_shape,
        output_shape,
        topology,
        uncertainty=False
    )

    # Check all (!) instance parameters
    assert model._topology == topology
    assert model._input_shape == X.shape[:2]
    assert model._output_shape == (21, 2)
    assert model._rnd_init == 'glorot_normal'
    assert not model._uncertainty
    assert model._dropout_rate == 0.1


def test_concat_concat(synthetic_data):
    # Get the data
    X = synthetic_data

    topology = [
        {
            'meta': {
                'layer_type': 'GRU',
                'layer_id': 'gru1',
                'parent_ids': ['input']},
            'params': {
                'units': 128,
                'return_sequences': False
            }
        },
        {
            'meta': {
                'layer_type': 'GRU',
                'layer_id': 'gru2',
                'parent_ids': ['input']
            },
            'params': {
                'units': 128,
                'return_sequences': False
            }
        },
        {
            'meta': {
                'layer_type': 'Concatenate',
                'layer_id': 'concat1',
                'parent_ids': ['gru1', 'gru2']
            },
            'params': {
                'axis': -1
            }
        }
    ]

    # Instantiate model
    input_shape = X.shape[:2]
    output_shape = (21, 10)
    model = SharedLayerModel(
        input_shape,
        output_shape,
        topology,
        uncertainty=True
    )

    # Check all (!) instance parameters
    assert model._topology == topology
    assert model._input_shape == X.shape[:2]
    assert model._output_shape == (21, 10)
    assert model._rnd_init == 'glorot_normal'
    assert model._uncertainty
    assert model._dropout_rate == 0.1
