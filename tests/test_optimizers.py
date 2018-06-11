# -*- coding: utf-8 -*-
import pytest

from fixtures import sample_data
from deep4cast import Forecaster, HyperOptimizer


def test_optimizer_init(sample_data):
    topology = [
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
    domain = {
        'lr': [0.001, 0.5]
    }
    forecaster = Forecaster(
        topology=topology,
        optimizer='rmsprop',
        lag=20,
        horizon=1,
        batch_size=8,
        epochs=50,
        lr=0.01
    )
    optimizer = HyperOptimizer(
        sample_data,
        forecaster,
        domain,
        n_iter=1
    )

    assert optimizer.domain == domain
    assert optimizer.n_iter == 1


def test_optimizer_fit(sample_data):
    topology = [
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
    domain = {
        'forecaster': {
            'epochs': (1, 20)
        },
        'topology': {
            'g2_units': (32, 128)
        },
        'optimizer': {
            'lr': (0.001, 0.5)
        },
    }
    forecaster = Forecaster(
        topology=topology,
        optimizer='rmsprop',
        lag=20,
        horizon=1,
        batch_size=8,
        epochs=50,
        lr=0.01
    )
    optimizer = HyperOptimizer(
        forecaster,
        sample_data,
        domain,
        n_iter=10
    )
    best_parameters, trials = optimizer.fit()
