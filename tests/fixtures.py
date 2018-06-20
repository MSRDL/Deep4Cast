# -*- coding: utf-8 -*-

import pytest

import numpy as np
from pandas import read_table


@pytest.fixture(scope='module')
def synthetic_data():
    N = 151
    M = 21
    L = 5
    return np.random.randn(N, M, L)


@pytest.fixture(scope='module')
def sample_data():
    # Load the data from file
    filename = './tests/timeseries_data.csv'
    try:
        df = read_table(filename, sep=',')
    except:
        print('Error: Run from tests from deep4cast folder.')
    ts = df.astype('float32').values
    ts = np.expand_dims(ts, axis=-1)

    return ts
