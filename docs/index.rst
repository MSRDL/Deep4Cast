=======================
Deep4Cast Documentation
=======================

Decision making incorporating uncertainty.
==========================================

**This package is under active development. Things may change :-).**

``Deep4Cast`` is a scalable machine learning package implemented in ``Python`` and ``Torch``. It has a front-end API similar to ``scikit-learn``. It is designed for medium to large time series data sets and allows for modeling of forecast uncertainties.

The network architecture is based on DeepMind's WaveNet. Regularization and approximate sampling from posterior predictive distributions of forecasts are achieved via Concrete Dropout.

Tutorials
---------

:ref:`/examples/github_forecasting.ipynb`

Main Requirements
-----------------

- `python 3.6 <http://python.org>`_
- `pytorch 1.0 <http://pytorch.org>`_

Installation
------------

From the package directory install the requirements and then the package (best in a clean virtual environment)

.. code-block::

    $ pip install -r requirements.txt
    $ python setup.py install

Authors
-------
- `Toby Bischoff <http://github.com/bischtob>`_
- Austin Gross
- Shirley Ren 
- `Kenneth Tran <http://www.kentran.net>`_

References
----------
- `Concrete Dropout <https://arxiv.org/pdf/1705.07832.pdf>`_ is used for approximate posterior Bayesian inference.
- `Wavenet <https://arxiv.org/pdf/1609.03499.pdf>`_ is used as encoder network.


.. toctree::
    :caption: Deep4Cast Documentation
    :maxdepth: 2

    dataset
    transforms
    models
    forecasters
    .. layers
    metrics

