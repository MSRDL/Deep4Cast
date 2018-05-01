# -*- coding: utf-8 -*-
"""Optimizer module for optimizing the hyperparameters of forecasters.

This module provides access to hyper-optimizer class that can be used to find
a nearly optimial set of hyperparamters for a given forecaster class.

"""

from functools import partial
from hyperopt import hp, fmin, tpe, space_eval, Trials
from inspect import getargspec
from .validators import TemporalCrossValidator

# Set the optimizable arguments globally and assign a hyperopt probability
# distribution to each of them.
__ALLOWED_TOPOLOGY_ARGS = {
    'activation': hp.choice,
    'use_bias': hp.choice,
    'activity_regularizer': hp.choice,
    'bias_initializer': hp.choice,
    'bias_regularizer': hp.choice,
    'bias_constraint': hp.choice,
    'kernel_initializer': hp.choice,
    'kernel_regularizer': hp.choice,
    'kernel_constraint': hp.choice,
    'recurrent_regularizer': hp.choice,
    'recurrent_constraint': hp.choice,
    'recurrent_dropout': hp.uniform,
    'depthwise_initializer': hp.choice,
    'pointwise_initializer': hp.choice,
    'rate': hp.uniform,
    'l1': hp.loguniform,
    'l2': hp.loguniform,
    'filters': partial(hp.quniform, q=1.0),
    'kernel_size': partial(hp.quniform, q=1.0),
    'strides': partial(hp.quniform, q=1.0),
    'dilation_rate': partial(hp.quniform, q=1.0),
    'cropping': partial(hp.quniform, q=1.0),
    'size': partial(hp.quniform, q=1.0),
    'padding': partial(hp.quniform, q=1.0),
    'pool_size': partial(hp.quniform, q=1.0),
    'cell': hp.choice,
    'units': partial(hp.quniform, q=1.0),
    'dropout': hp.loguniform,
    'alpha': hp.loguniform,
    'alpha_initializer': hp.choice,
    'alpha_regularizer': hp.choice,
    'alpha_constraint': hp.choice,
    'theta': hp.loguniform,
    'stddev': hp.loguniform
}

__ALLOWED_OPTIMIZER_ARGS = {
    'lr': hp.loguniform,
    'momentum': hp.loguniform,
    'decay': hp.loguniform,
    'nesterov': hp.choice,
    'rho': hp.loguniform,
    'epsilon': hp.loguniform,
    'beta_1': hp.loguniform,
    'beta_2': hp.loguniform,
    'amsgrad': hp.choice,
    'schedule_decay': hp.loguniform
}

__ALLOWED_FORECASTER_ARGS = {
    'lookback': partial(hp.quniform, q=1.0),
    'batch_size': partial(hp.quniform, q=1.0),
    'nepochs': partial(hp.quniform, q=1.0),
}


class HyperOptimizer():
    """Hyper-parameter optimizer class based on the hyperopt package.
    :param data: Dataset to optimize on.
    :type data: numpy array
    :param forecaster: Forecaster to optimize.
    :type forecaster: Forecaster instance
    :param domain: Parameter domain to explore by optimizer.
    :type domain: dict
    :param niter: Optimization budget in number of iterations.
    :type niter: int

    """

    def __init__(self, data, forecaster, domain, niter=10, **kwargs):
        """Initialize properties."""
        self.data = data
        self.forecaster = forecaster
        self.domain = domain
        self.niter = niter
        self.train_frac = 0.7
        self.nfolds = 3
        self.loss = 'mse'
        allowed_args = ('train_frac', 'nfolds', 'loss')
        for arg, value in kwargs.items():
            if arg in allowed_args:
                setattr(self, arg, value)
            else:
                raise ValueError('Invalid keyword argument: {}.'.format(arg))
        self.validator = TemporalCrossValidator

    def fit(self):
        """Fit the model hyperparameters to the data."""
        objective = self._create_objective
        space = self._create_hyperparameter_domain()

        # Useing the Trials object allows us to keep track of every trial.
        trials = hp.Trials()
        best_parameters = hp.fmin(
            fn=objective,
            space=space,
            algo=self.algorithm,
            trials=trials,
            max_evals=self.niter
        )

        return best_parameters, trials

    def _create_objective(self):
        """Return the objective function to be optimized."""
        
        def objective(**kwargs):
            """Objective function closure to be optimized by 
            hyperparameter optimization framework.
            """

            #
            optimizer_params = getargspec(self.forecaster.optimizer)
            topology_params = self.get_topol_params(self.forecaster.topology)
            forecaster_params = ('lookback', 'batch_size', 'nepochs')

            #
            for key, value in kwargs.items():
                if key in optimizer_params:
                    setattr(self.forecaster.optimizer, key, value)
                elif key in topology_params:
                    setattr(self.forecaster.topology, key, value)
                elif key in forecaster_params:
                    setattr(self.forecaster, key, value)
                else:
                    raise ValueError(
                        '{} not a forecaster parameter.'.format(key)
                    )

            # Use the validator attribute to generate a score for
            # a set of input parameters.
            validator = self.validator(
                self.data,
                self.forecaster,
                self.train_frac,
                self.nfolds,
                self.loss
            )
            score = validator.evaluate(verbose=False)

            return score

        return objective

    def _create_hyperparameter_domain(self):
        """Return the hyper-parameter domain needed for hyperopt package."""
        parameter_domain = []

        # Find optimizable parameters of forecaster topology from
        # forecaster attribute.
        optimizer_params = getargspec(self.forecaster.optimizer)
        topology_params = self.get_topol_params(self.forecaster.topology)
        forecaster_params = ('lookback', 'batch_size', 'nepochs')

        # Generate hyper-optmizer parameter_domain for optimizer parameters
        # but make sure that parameters are ordered and are allowed to
        # be optimized.
        for key, value in self.domain['optimizer']:
            low, high = value
            if low > high:
                raise ValueError('Parameter range must be [low, high].')
            if key in optimizer_params and key in __ALLOWED_OPTIMIZER_ARGS:
                parameter_domain.append(
                    __ALLOWED_OPTIMIZER_ARGS[key](
                        label=key,
                        low=low,
                        high=high
                    )
                )
            else:
                raise ValueError('{} not an optimizer parameter.'.format(key))

        # Same as above. Generate hyper-optmizer parameter_domain for topology
        # parameters. For the topology object, more logic is required to
        # distringuish different layers and to separate out the layer
        # activations.
        for key, value in self.domain['topology']:
            param = '_'.join(key.split()[1:])
            low, high = value
            if low > high:
                raise ValueError('Parameter range must be [low, high].')
            if param in topology_params and param in __ALLOWED_TOPOLOGY_ARGS:
                # Activation functions are handled the same for all layers
                if param == 'activation':
                    label = param
                else:
                    label = key
                parameter_domain.append(
                    __ALLOWED_TOPOLOGY_ARGS[param](
                        label=label,
                        low=low,
                        high=high
                    )
                )
            else:
                raise ValueError('{} not a topology parameter.'.format(param))

        # Same as above. Generate hyper-optmizer parameter_domain for topology
        # parameters.        for key, value in self.domain['forecaster']:
            low, high = value
            if low > high:
                raise ValueError('Parameter range must be [low, high].')
            if key in forecaster_params and key in __ALLOWED_FORECASTER_ARGS:
                parameter_domain.append(
                    __ALLOWED_FORECASTER_ARGS[key](
                        label=key,
                        low=low,
                        high=high
                    )
                )
            else:
                raise ValueError('{} not a forecaster parameter.'.format(key))

        return parameter_domain

    @staticmethod
    def _get_topol_params(topology):
        """Return a list of all topology parameters with None value.
        This is necessary for construction of the hyper-optimizer objective
        function because the objective's parameters may be topology parameters.

        We return all None-valued parameters as independently optimizable
        parameters, except for the layer activations.
        """
        params = set()
        for layer in topology:
            layer_id = layer['id']
            for key, value in layer['params'].items():
                if value is None:
                    if key is not 'activation':
                        params.add(layer_id + '_' + key)
                    else:
                        params.add(key)
        params = list(params)

        return params

    @staticmethod
    def _set_topol_param(topology, param, value):
        """Set topology parameters IN PLACE (!)."""
        if param in __ALLOWED_TOPOLOGY_ARGS.keys():
            layer_id = param.split()[0]
            layer_param = '_'.join(param.split()[1:])
            for i, layer in enumerate(topology):
                if layer_id == layer['id']:
                    layer_ind = i
                    break
            else:
                raise KeyError('Topology has no parameter {}'.format(param))

            topology[layer_ind]['params'][layer_param] = value
