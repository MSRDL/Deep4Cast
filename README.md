# Deep4cast: Forecasting for Decision Making under Uncertainty

<img src="https://raw.githubusercontent.com/MSRDL/Deep4Cast/master/docs/images/thumb.jpg" height=200>

***This package is under active development. Things may change :-).***

``Deep4Cast`` is a scalable machine learning package implemented in ``Python`` and ``Torch``. It has a front-end API similar to ``scikit-learn``. It is designed for medium to large time series data sets and allows for modeling of forecast uncertainties.

The network architecture is based on ``WaveNet``. Regularization and approximate sampling from posterior predictive distributions of forecasts are achieved via ``Concrete Dropout``.

Documentation is available at [read the docs](https://deep4cast.readthedocs.io/en/latest/).

## Installation

### Main Requirements
- [python](http://python.org) - version 3.6
- [pytorch](http://pytorch.org) - version 1.0

### Source
Before installing we recommend setting up a clean [virtual environment](https://docs.python.org/3.6/tutorial/venv.html).

From the package directory install the requirements and then the package.
```
$ pip install -r requirements.txt
$ python setup.py install
```

## Examples
- [Tutorial Notebooks](https://github.com/MSRDL/Deep4Cast/blob/master/docs/examples)

## Authors: 
- [Toby Bischoff](http://github.com/bischtob)
- Austin Gross
- [Kenneth Tran](http://www.kentran.net)

## References:
- [Concrete Dropout](https://arxiv.org/pdf/1705.07832.pdf) is used for approximate posterior Bayesian inference.
- [Wavenet](https://arxiv.org/pdf/1609.03499.pdf) is used as encoder network.
