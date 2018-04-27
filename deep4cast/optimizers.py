# -*- coding: utf-8 -*-
"""Optimizer module for optimizing the hyperparameters of forecasters.

This module provides access to hyper-optimizer class that can be used to find
a nearly optimial set of hyperparamters for a given model class.

"""

    # "sweeper = random_sweep(\n",
    # "    train_features, train_labels, \n",
    # "    XGBClassifier(), {'max_depth': [3, 5, 10, 20], \n",
    # "                      'n_estimators': [10, 20, 40, 80],\n",
    # "                      'subsample': [1, 0.8]},\n",
    # "    scoring=pr_scorer, n_iter=10)

import numpy as numpy

from . import validators
from functools import partial
from hyperopt import hp, fmin, tpe, space_eval

class Optimizer():

    optimizer = None

    def __init__(self, algorithm=None):

        self.algorithm = algorithm or tpe.suggest
        self.optimizer = fmin

    def report(self):
        pass


