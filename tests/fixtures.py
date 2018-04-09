# -*- coding: utf-8 -*-

import pytest

import numpy as np


@pytest.fixture(scope='module')
def synthetic_data():
    N = 150
    M = 25
    return np.random.randn(N, M)
