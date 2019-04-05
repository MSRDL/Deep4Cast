Layers
======

Concete Dropout
---------------

Dropout randomly sets a fraction of input units to 0
at each update during training time, which helps prevent overfitting. At
prediction time the units are then also dropped out with the same fraction.
This generates samples from an approximate posterior predictive
distribution. Unlike in MCDropout, in Concrete Dropout the dropout rates
are learned from the data. This version of Concrete Dropout cannot be used 
with additional regularization.

References:
    - `Dropout: A Simple Way to Prevent Neural Networks from Overfitting <http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf>`_
    - `MCDropout: Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning <https://arxiv.org/abs/1506.02142>`_
    - `Concrete Dropout <https://papers.nips.cc/paper/6949-concrete-dropout.pdf>`_

.. automodule:: custom_layers
  :members: