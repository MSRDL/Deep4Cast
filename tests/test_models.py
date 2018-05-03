# -*- coding: utf-8 -*-

from fixtures import synthetic_data

from deep4cast.models import LayeredTimeSeriesModel


def test_init_GRU(synthetic_data):
    X = synthetic_data
    topology = [
        ('GRU', {'units': 128, 'activation': 'tanh'}),
        ('GRU', {'units': 35, 'activation': 'tanh'})
    ]
    model = LayeredTimeSeriesModel(X.shape, topology)

    assert model._topology == topology
    assert model._input_shape == synthetic_data.shape
    assert model._rnd_init == 'glorot_normal'
    assert len(model.layers) == len(topology) + 1 + 1


def test_init_LSTM(synthetic_data):
    X = synthetic_data
    topology = [
        ('LSTM', {'units': 12, 'activation': 'relu'}),
        ('LSTM', {'units': 127, 'activation': 'relu'}),
        ('LSTM', {'units': 127, 'activation': 'relu'}),
        ('LSTM', {'units': 5, 'activation': 'relu'})
    ]
    model = LayeredTimeSeriesModel(X.shape, topology)

    assert model._topology == topology
    assert model._input_shape == synthetic_data.shape
    assert model._rnd_init == 'glorot_normal'
    assert len(model.layers) == len(topology) + 1 + 1


def test_init_Conv1D_deep(synthetic_data):
    X = synthetic_data
    topology = [
        ('Conv1D', {'filters': 64, 'kernel_size': 5, 'activation': 'relu'}),
        ('MaxPooling1D', {'pool_size': 3, 'strides': 1}),
        ('Conv1D', {'filters': 67, 'kernel_size': 4, 'activation': 'relu'}),
        ('MaxPooling1D', {'pool_size': 3, 'strides': 1}),
        ('Conv1D', {'filters': 128, 'kernel_size': 6, 'activation': 'relu'}),
        ('MaxPooling1D', {'pool_size': 2, 'strides': 2}),
        ('Flatten', {}),
        ('Dense', {'units': 128})
    ]
    model = LayeredTimeSeriesModel(X.shape, topology)

    assert model._topology == topology
    assert model._input_shape == synthetic_data.shape
    assert model._rnd_init == 'glorot_normal'
    assert len(model.layers) == len(topology) + 1 + 1


def test_init_LSTM_Conv1D(synthetic_data):
    X = synthetic_data
    topology = [
        ('LSTM', {'units': 127}),
        ('Conv1D', {'filters': 64, 'kernel_size': 5, 'activation': 'elu'}),
        ('MaxPooling1D', {'pool_size': 3, 'strides': 1}),
        ('Conv1D', {'filters': 67, 'kernel_size': 4, 'activation': 'elu'}),
        ('MaxPooling1D', {'pool_size': 3, 'strides': 1}),
        ('Conv1D', {'filters': 128, 'kernel_size': 6, 'activation': 'elu'}),
        ('MaxPooling1D', {'pool_size': 2, 'strides': 2}),
        ('Flatten', {}),
        ('Dense', {'units': 128})
    ]
    model = LayeredTimeSeriesModel(X.shape, topology)

    assert model._topology == topology
    assert model._input_shape == synthetic_data.shape
    assert model._rnd_init == 'glorot_normal'
    assert len(model.layers) == len(topology) + 1 + 1


def test_init_Conv1D_GRU(synthetic_data):
    X = synthetic_data
    topology = [
        ('Conv1D', {'filters': 64, 'kernel_size': 5, 'activation': 'elu'}),
        ('GRU', {'units': 128, 'activation': 'tanh'}),
    ]
    model = LayeredTimeSeriesModel(X.shape, topology)

    assert model._topology == topology
    assert model._input_shape == synthetic_data.shape
    assert model._rnd_init == 'glorot_normal'
    assert len(model.layers) == len(topology) + 1 + 1


def test_init_Empty(synthetic_data):
    X = synthetic_data
    model = LayeredTimeSeriesModel(X.shape)

    assert not model._topology
    assert model._input_shape == synthetic_data.shape
    assert model._rnd_init == 'glorot_normal'
    assert len(model.layers) == 1
