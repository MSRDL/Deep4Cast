# -*- coding: utf-8 -*-

import pytest

import numpy as np


@pytest.fixture(scope='module')
def synthetic_data():
    N = 151
    M = 21
    L = 5
    return np.random.randn(N, M, L)
