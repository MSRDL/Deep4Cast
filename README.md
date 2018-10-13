@Toby:
- Almost never check in binary files
- Code that is outside the scope of the library (e.g. `process_blah.py` should be in the example notebooks). Personal code should have `personal` in the filename so that it's ignored by git.
- `requirements.txt` contained too many packages. It should contain a minimal set of required packages.

# Deep4cast

***This package is under development.***

```Deep4cast``` provides easy access to deep learning-based tools for forecasting, anomaly detection, and counterfactual analysis.

### Forecasting
Quickstart tutorial can be found in ```/tutorials/quickstart.ipynb```.

### Anomaly Detection
Tutorial coming soon.

## Documentation
Documentation can be found in ```/docs/build/html/index.html```.

## Installation
### Source
From the package directory you then need to install the requirements and the package
```
$ pip install -r requirements.txt
$ python setup.py install
```
or with pip's editable model (useful for development) as
```
$ pip install -r requirements.txt
$ pip install -e .
```

### Main Requirements
```
Python 3.6
Tensorflow, CTNK, or Theano
Keras
```

## Contributor Instructions
Coming soon

## Authors
TBD
