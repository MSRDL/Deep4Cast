# -*- coding: utf-8 -*-
import pytest

from keras.optimizers import SGD, RMSprop

from fixtures import synthetic_data
from deep4cast import Forecaster
from deep4cast.models import SharedLayerModel


def test_defaults():
    topology = [
        {
            'meta': {
                'layer_type': 'GRU',
                'layer_id': 'gru1',
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
                'layer_id': 'gru2',
                'parent_ids': ['gru1']
            },
            'params': {
                'units': 128,
                'return_sequences': False
            }
        }
    ]
    model = Forecaster(
        topology,
        optimizer='sgd',
        lag=10,
        horizon=1,
        batch_size=8,
        epochs=50,
        lr=0.01,
        nesterov=True
    )

    # Check all (!) instance parameters
    assert model.model_class == SharedLayerModel
    assert model.topology == topology
    assert model.uncertainty is False
    assert model.dropout_rate == 0.1
    assert model._model is None
    assert model.optimizer == 'SGD'
    assert model.lag == 10
    assert model.horizon == 1
    assert model.batch_size == 8
    assert model.epochs == 50
    assert model.history is None
    assert model.loss == 'mse'
    assert model.metrics == ['mape']
    assert model.seed is None
    assert model._data_means is None
    assert model._data_scales is None
    assert model._is_fitted is False
    assert model._is_standardized is False
    assert isinstance(model._optimizer, SGD)

    # Check optimizer args
    assert model._optimizer.lr == 0.01
    assert model._optimizer.nesterov


def test_defaults_fit(synthetic_data):
    # Get the data
    X = synthetic_data

    topology = [
        {
            'meta': {
                'layer_type': 'GRU',
                'layer_id': 'gru1',
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
                'layer_id': 'gru2',
                'parent_ids': ['gru1']
            },
            'params': {
                'units': 128,
                'return_sequences': False
            }
        }
    ]
    model = Forecaster(
        topology,
        optimizer='sgd',
        lag=10,
        horizon=1,
        batch_size=8,
        epochs=50,
        lr=0.01,
        nesterov=True
    )
    model.fit(X)
    y_pred = model.predict(X)

    # Check all (!) instance parameters
    assert model.model_class == SharedLayerModel
    assert model.topology == topology
    assert model.uncertainty is False
    assert model.dropout_rate == 0.1
    assert model._model is not None
    assert model.optimizer == 'SGD'
    assert model.lag == 10
    assert model.horizon == 1
    assert model.batch_size == 8
    assert model.epochs == 50
    assert model.history is not None
    assert model.loss == 'mse'
    assert model.metrics == ['mape']
    assert model.seed is None
    assert model._data_means is not None
    assert model._data_scales is not None
    assert model._is_fitted is True
    assert model._is_standardized is True
    assert isinstance(model._optimizer, SGD)

    # Check optimizer args
    assert model._optimizer.lr == 0.01
    assert model._optimizer.nesterov


def test_defaults_fit_mcdropout(synthetic_data):
    # Get the data
    X = synthetic_data

    topology = [
        {
            'meta': {
                'layer_type': 'GRU',
                'layer_id': 'gru1',
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
                'layer_id': 'gru2',
                'parent_ids': ['gru1']
            },
            'params': {
                'units': 128,
                'return_sequences': False
            }
        }
    ]
    model = Forecaster(
        topology,
        optimizer='rmsprop',
        lag=10,
        horizon=2,
        batch_size=8,
        epochs=50,
        uncertainty=True,
        lr=0.1
    )
    model.fit(X)
    y_pred = model.predict(X)

    # Check all (!) instance parameters
    assert model.model_class == SharedLayerModel
    assert model.topology == topology
    assert model.uncertainty is True
    assert model.dropout_rate == 0.1
    assert model._model is not None
    assert model.optimizer == 'RMSprop'
    assert model.lag == 10
    assert model.horizon == 2
    assert model.batch_size == 8
    assert model.epochs == 50
    assert model.history is not None
    assert model.loss == 'mse'
    assert model.metrics == ['mape']
    assert model.seed is None
    assert model._data_means is not None
    assert model._data_scales is not None
    assert model._is_fitted is True
    assert model._is_standardized is True
    assert isinstance(model._optimizer, RMSprop)

    # Check optimizer args
    assert model._optimizer.lr == 0.1


def test_concat_fit(synthetic_data):
    # Get the data
    X = synthetic_data

    topology = [
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
    model = Forecaster(
        topology,
        optimizer='sgd',
        lag=10,
        horizon=5,
        batch_size=8,
        epochs=50,
        uncertainty=True,
        lr=0.01,
        nesterov=True
    )
    model.fit(X)

    # Check all (!) instance parameters
    assert model.model_class == SharedLayerModel
    assert model.topology == topology
    assert model.uncertainty is True
    assert model.dropout_rate == 0.1
    assert model._model is not None
    assert model.optimizer == 'SGD'
    assert model.lag == 10
    assert model.horizon == 5
    assert model.batch_size == 8
    assert model.epochs == 50
    assert model.history is not None
    assert model.loss == 'mse'
    assert model.metrics == ['mape']
    assert model.seed is None
    assert model._data_means is not None
    assert model._data_scales is not None
    assert model._is_fitted is True
    assert model._is_standardized is True
    assert isinstance(model._optimizer, SGD)
