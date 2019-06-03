=======================
Deep4Cast Documentation
=======================

Forecasting for decision making under uncertainty
=================================================

**This package is under active development. Things may change :-).**

``Deep4Cast`` is a scalable machine learning package implemented in ``Python`` and ``Torch``. It has a front-end API similar to ``scikit-learn``. It is designed for medium to large time series data sets and allows for modeling of forecast uncertainties.

The network architecture is based on ``WaveNet``. Regularization and approximate sampling from posterior predictive distributions of forecasts are achieved via ``Concrete Dropout``.

Examples
--------

:ref:`/examples/github_forecasting.ipynb`

Benchmark performance
---------------------
- M4 forecasting competition dataset

Authors
-------
- `Toby Bischoff <http://github.com/bischtob>`_
- Austin Gross
- `Kenneth Tran <http://www.kentran.net>`_


References
----------
- `Concrete Dropout <https://arxiv.org/pdf/1705.07832.pdf>`_ is used for approximate posterior Bayesian inference.
- `Wavenet <https://arxiv.org/pdf/1609.03499.pdf>`_ is used as encoder network.


.. toctree::
    :maxdepth: 2
    :glob:
    :hidden:

    get_started
    examples/*
    datasets
    transforms
    models
    forecasters
    metrics
    custom_layers

