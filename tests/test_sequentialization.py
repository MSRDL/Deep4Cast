# -*- coding: utf-8 -*-
import numpy as np
import pytest

from deep4cast import Forecaster


def test_single_covariate_single_example():
    data = np.random.randn(1, 123, 1)
    forecaster = Forecaster(
        topology='gru',
        optimizer='sgd',
        lookback=13,
        horizon=1,
        batch_size=8,
        epochs=50,
        lr=0.01
    )
    X, y = forecaster._sequentialize(data)
    assert X.shape == (111, 13, 1)
    assert y.shape == (111, 1, 1)


def test_single_covariate_multiple_example():
    data = np.random.randn(2, 123, 1)
    forecaster = Forecaster(
        topology='gru',
        optimizer='sgd',
        lookback=13,
        horizon=3,
        batch_size=8,
        epochs=50,
        lr=0.01
    )
    X, y = forecaster._sequentialize(data)
    assert X.shape == (222, 13, 1)
    assert y.shape == (111 * 2, 3, 1)


def test_multiple_covariate_single_examples():
    data = np.random.randn(1, 123, 34)
    forecaster = Forecaster(
        topology='gru',
        optimizer='sgd',
        lookback=13,
        horizon=5,
        batch_size=8,
        epochs=50,
        lr=0.01
    )
    X, y = forecaster._sequentialize(data)
    assert X.shape == (111, 13, 34)
    assert y.shape == (111, 5, 34)


def test_multiple_covariate_multiple_examples():
    data = np.random.randn(23, 123, 34)
    forecaster = Forecaster(
        topology='gru',
        optimizer='sgd',
        lookback=13,
        horizon=1,
        batch_size=8,
        epochs=50,
        lr=0.01
    )
    X, y = forecaster._sequentialize(data)
    assert X.shape == (111 * 23, 13, 34)
    assert y.shape == (111 * 23, 1, 34)


def test_single_covariate_multiple_unequal_examples():
    data = [
        np.random.randn(123, 1),
        np.random.randn(24, 1),
        np.random.randn(234, 1)
    ]
    data = np.array(data)
    forecaster = Forecaster(
        topology='gru',
        optimizer='sgd',
        lookback=13,
        horizon=3,
        batch_size=8,
        epochs=50,
        lr=0.01
    )
    X, y = forecaster._sequentialize(data)
    assert X.shape == (111 + 12 + 222, 13, 1)
    assert y.shape == (111 + 12 + 222, 3, 1)


def test_mutiple_covariate_multiple_unequal_examples():
    data = [
        np.random.randn(123, 34),
        np.random.randn(24, 34),
        np.random.randn(234, 34)
    ]
    data = np.array(data)
    forecaster = Forecaster(
        topology='gru',
        optimizer='sgd',
        lookback=13,
        horizon=3,
        batch_size=8,
        epochs=50,
        lr=0.01
    )
    X, y = forecaster._sequentialize(data)
    assert X.shape == (111 + 12 + 222, 13, 34)
    assert y.shape == (111 + 12 + 222, 3, 34)


def test_mutiple_covariate_multiple_unequal_examples_with_targets():
    data = [
        np.random.randn(123, 34),
        np.random.randn(24, 34),
        np.random.randn(234, 34)
    ]
    data = np.array(data)
    forecaster = Forecaster(
        topology='gru',
        optimizer='sgd',
        lookback=13,
        horizon=3,
        batch_size=8,
        epochs=50,
        lr=0.01
    )
    forecaster.targets = [0, 2, 3, 6]
    X, y = forecaster._sequentialize(data)
    assert X.shape == (111 + 12 + 222, 13, 34)
    assert y.shape == (111 + 12 + 222, 3, 4)


def test_mutiple_covariate_multiple_examples_with_targets_for_predict():
    data = np.random.randn(3, 13, 34)
    forecaster = Forecaster(
        topology='gru',
        optimizer='sgd',
        lookback=13,
        horizon=3,
        batch_size=8,
        epochs=50,
        lr=0.01
    )
    forecaster.targets = [0, 2, 3, 6]
    X, y = forecaster._sequentialize(data)
    assert X.shape == (3, 13, 34)
    assert y.shape == (3, 3, 4)
