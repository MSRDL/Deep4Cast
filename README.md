## Deep4cast: Forecasting for Decision Making under Uncertainty

<img src="https://raw.githubusercontent.com/MSRDL/Deep4Cast/pytorch/doc/images/thumb.jpg" height=200>

***This package is under active development. Things may change :-).***

```Deep4Cast``` is a scalable machine learning package implemented in Python. It has a front-end API designed to be similar to scikit-learn but is based on Pytorch. It was designed for medium to large size time series data sets and allows for modeling of forecast uncertainties. The network architecture is based on DeepMind's WaveNet. Regularization and approximate sampling from posterior predictive distributions of forecasts are achieved via Concrete Dropout.

Package documentation under construction. Please see example notebooks for instructions.

## Installation
### Source
From the package directory install the requirements and then the package (best in a clean virtual environment)
```
$ pip install -r requirements.txt
$ python setup.py install
```

### Main Requirements
- [python](http://python.org) - version 3.6
- [pytorch](http://pytorch.org) - version 1.0

## Examples
- [Tutorial Notebooks](https://github.com/MSRDL/Deep4Cast/blob/master/examples)

## Authors: 
- [Toby Bischoff](http://github.com/bischtob)
- Austin Gross
- Shirley Ren 
- [Kenneth Tran](http://www.kentran.net)

## References:
- [Concrete Dropout](https://arxiv.org/pdf/1705.07832.pdf) is used for approximate posterior Bayesian inference.
- [Wavenet](https://arxiv.org/pdf/1609.03499.pdf) is used as encoder network.
