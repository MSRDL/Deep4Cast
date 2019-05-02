Transformations
===============

Transformations of the time series intended to be used in a similar fashion to `torchvision <https://pytorch.org/docs/stable/torchvision/transforms.html>`_. The user specifies a dictionary of Transformations in a particular order.

Example::

    >>> transform = [
    >>>     {'Tensorize': None},
    >>>     {'LogTransform': {'targets': [0], 'offset': 1.0}},
    >>>     {'RemoveLast': {'targets': [0]}},
    >>>     {'Target': {'targets': [0]}}
    >>> ]

.. automodule:: transforms
  :members: