# -*- coding: utf-8 -*-
import pytest

from fixtures import synthetic_data
from deep4cast import CNNForecaster, RNNForecaster


def test_RNN_defaults():
    model = RNNForecaster()

    assert model._epochs == 50
    assert len(model._topology) == 1
    assert model._learning_rate == 0.01
    assert model.history is None
    assert model._data_std is None
    assert model._is_fitted is False


def test_RNN_custom_topology(synthetic_data):
    topology = [('GRU', {'units': 32}),
                ('GRU', {'units': 32, 'activation': 'relu'})]
    model = RNNForecaster(topology)

    assert model._epochs == 50
    assert len(model._topology) == 2
    assert model._learning_rate == 0.01
    assert model.history is None
    assert model._data_std is None
    assert model._is_fitted is False


def test_RNN_fitting(synthetic_data):
    X = synthetic_data
    topology = [('GRU', {'units': 32}),
                ('GRU', {'units': 32, 'activation': 'relu'})]
    model = RNNForecaster(topology)
    model.fit(X, lookback_period=4)

    assert model._epochs == 50
    assert len(model._topology) == 2
    assert model._learning_rate == 0.01
    assert model._is_fitted is True


def test_CNN_defaults():
    model = CNNForecaster()

    assert model._epochs == 50
    assert len(model._topology) == 7
    assert model._learning_rate == 0.1
    assert model.history is None
    assert model._data_std is None
    assert model._is_fitted is False


def test_CNN_custom_topology(synthetic_data):
    topology = [
        ('Conv1D', {'filters': 64, 'kernel_size': 5, 'activation': 'elu'}),
        ('MaxPooling1D', {'pool_size': 3, 'strides': 1}),
        ('Conv1D', {'filters': 128, 'kernel_size': 3, 'activation': 'elu'}),
        ('MaxPooling1D', {'pool_size': 1, 'strides': 1}),
        ('Flatten', {}),
        ('Dense', {'units': 128})
    ]
    model = CNNForecaster(topology)

    assert model._epochs == 50
    assert len(model._topology) == len(topology)
    assert model._learning_rate == 0.1
    assert model.history is None
    assert model._data_std is None
    assert model._is_fitted is False


def test_CNN_fitting(synthetic_data):
    X = synthetic_data
    topology = [
        ('Conv1D', {'filters': 64, 'kernel_size': 5, 'activation': 'elu'}),
        ('MaxPooling1D', {'pool_size': 3, 'strides': 1}),
        ('Conv1D', {'filters': 63, 'kernel_size': 4, 'activation': 'elu'}),
        ('MaxPooling1D', {'pool_size': 4, 'strides': 2}),
        ('Conv1D', {'filters': 111, 'kernel_size': 3, 'activation': 'elu'}),
        ('MaxPooling1D', {'pool_size': 1, 'strides': 1}),
        ('Flatten', {}),
        ('Dense', {'units': 127})
    ]
    model = CNNForecaster(topology)
    model.fit(X, lookback_period=20)

    assert model._epochs == 50
    assert len(model._topology) == len(topology)
    assert model._learning_rate == 0.1
    assert model._is_fitted is True


def test_CNN_GRU_fitting(synthetic_data):
    X = synthetic_data
    topology = [
        ('Conv1D', {'filters': 64, 'kernel_size': 5, 'activation': 'relu'}),
        ('GRU', {'units': 32, 'activation': 'relu'})
    ]
    model = CNNForecaster(topology)
    model.fit(X, lookback_period=20)

    assert model._epochs == 50
    assert len(model._topology) == len(topology)
    assert model._learning_rate == 0.1
    assert model._is_fitted is True
