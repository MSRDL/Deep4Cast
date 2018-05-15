# -*- coding: utf-8 -*-
import pytest

from fixtures import synthetic_data
from deep4cast import CNNForecaster, RNNForecaster, TemporalCrossValidator


def test_Validator1(synthetic_data):

    validator = TemporalCrossValidator(
        RNNForecaster,
        synthetic_data,
        train_frac=0.75,
        n_folds=1,
        loss='mape'
    )

    x = {
        'topology': [('GRU', {'units': 128})],
        'lookback': 20,
        'batch_size': 8,
        'epochs': 50,
        'learning_rate': 0.01
    }
    scores = validator(x)

    assert validator.model == RNNForecaster
    assert validator.loss == 'mape'
    assert scores


def test_Validator2(synthetic_data):

    validator = TemporalCrossValidator(
        RNNForecaster,
        synthetic_data,
        train_frac=0.75,
        n_folds=3,
        loss='smape'
    )

    x = {
        'topology': [('GRU', {'units': 128})],
        'lookback': 20,
        'batch_size': 8,
        'epochs': 50,
        'learning_rate': 0.01
    }
    scores = validator.evaluate(x)

    assert validator.model == RNNForecaster
    assert validator.loss == 'smape'
    assert scores
