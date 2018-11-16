"""Time series forecasting module.
This module provides access to forecasters that can be fit multivariate time
series given as numpy arrays.
"""

from inspect import getfullargspec
from typing import Union
import numpy as np
import keras.optimizers

from keras.callbacks import TerminateOnNaN
from . import loss_functions

class Forecaster():
    """Forecaster class.

    Take a keras model and builds a forecaster.

    """

    def __init__(self,
                 model: keras.Model,
                 loss: Union[str, loss_functions.Loss]='gaussian',
                 optimizer='adam',
                 batch_size=16,
                 epochs=100,
                 **kwargs):
        """Initialize properties."""
        # Neural network model attributes
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.history = None

        # Set up the loss function
        """Instantiate the loss."""
        if isinstance(loss, loss_functions.Loss):
            self.loss = loss
        elif isinstance(loss, str):
            self.loss = getattr(loss_functions, loss)(
                self.model.output_shape[1:]
            )
        else:
            raise ValueError('Not a valid loss function type.')

        # Set up optimizer
        self.optimizer = getattr(keras.optimizers, optimizer)()
        allowed_kwargs = self.get_optimizer_args()
        for key, value in kwargs.items():
            if key not in allowed_kwargs:
                raise ValueError('{} not a valid optimizer argument.'.format(key))
        self.set_optimizer_args(kwargs)

        # Boolean checks for refitting
        self.is_fitted = False

    def fit(self, X, y, verbose=0):
        """Fit model to data."""
        # Need to handle the case where the model is fitted for more epochs
        # after it has already been fitted
        if not self.is_fitted:
            # Keras needs to compile the computational graph before fitting
            self.model.compile(loss=self.loss, optimizer=self.optimizer)

        # Fit model to data
        self.history = self.model.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[TerminateOnNaN()],
            verbose=verbose
        )

        # Change state to fitted so that other methods work correctly
        self.is_fitted = True

    def predict(self, X, n_samples=1000):
        """Generate predictions for input time series numpy array.
        :param X: Time series array of shape (n_steps, n_variables).
        :type X: numpy.array
        :param n_samples: Number of prediction samples (>= 1).
        :type n_samples: int

        """
        # Check if model is actually fitted
        if not self.is_fitted:
            raise ValueError('The model has not been fitted.')

        # Repeat the prediction n_samples times to generate samples from
        # approximate posterior predictive distribution.
        block_size = len(X)
        X = np.repeat(X, [n_samples] * len(X), axis=0)

        # Make predictions for parameters of pdfs then sample from pdfs
        predictions = self.model.predict(X, self.batch_size)
        predictions = self.loss.sample(predictions, n_samples=1)

        # Reorganize prediction samples into 3D array
        reshuffled_predictions = []
        for i in range(block_size):
            block = predictions[i * n_samples:(i + 1) * n_samples]
            block = np.expand_dims(block, axis=1)
            reshuffled_predictions.append(block)
        predictions = np.concatenate(reshuffled_predictions, axis=1)

        return predictions

    def get_optimizer_args(self):
        """Get optimizer parameters."""
        args = getfullargspec(self.optimizer.__class__)[0]
        args.remove('self')
        return args

    def set_optimizer_args(self, params):
        """Set optimizer parameters."""
        optimizer_class = self.optimizer.__class__
        optimizer_args = getfullargspec(optimizer_class)[0]
        for key, value in params.items():
            if key in optimizer_args:
                setattr(self.optimizer, key, value)

    def reset(self):
        """Reset model for refitting."""
        self.is_fitted = False


class VectorScaler():
    """Scaler class.

    Rescale vectors removing mean and dividing by standard deviation
    on a component basis.

    :param targets: targets in the data that should be rescaled.
    :type targets: list

    """

    def __init__(self, targets=None):
        """Initialize properties."""
        self.targets = targets
        self.x_mean = None
        self.x_std = None
        self.xis_fitted = False
        self.y_mean = None
        self.y_std = None
        self.yis_fitted = False

    def fit_x(self, X):
        """Fit the scaler."""
        if self.targets is None:
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
        else:
            # Need to concatenate mean with zeros and stds with ones for
            # categorical targets
            mean = np.zeros(X.shape[1:])
            std = np.ones(X.shape[1:])
            mean[:, self.targets] = np.mean(X[:, :, self.targets], axis=0)
            std[:, self.targets] = np.std(X[:, :, self.targets], axis=0)

        self.x_mean = mean
        self.x_std = std
        self.xis_fitted = True

    def fit_y(self, y):
        """Fit the scaler."""
        self.y_mean = np.mean(y, axis=0)
        self.y_std = np.std(y, axis=0)
        self.yis_fitted = True

    def transform_x(self, X):
        return (X - self.x_mean) / self.x_std

    def transform_y(self, y):
        return (y - self.y_mean) / self.y_std

    def fit_transform_x(self, X):
        self.fit_x(X)
        return self.transform_x(X)

    def fit_transform_y(self, y):
        self.fit_y(y)
        return self.transform_y(y)

    def inverse_transform_x(self, X):
        if self.xis_fitted:
            return X * self.x_std + self.x_mean
        else:
            raise ValueError('Not fitted on X.')

    def inverse_transform_y(self, y):
        if self.yis_fitted:
            return y * self.y_std + self.y_mean
        else:
            raise ValueError('Not fitted on y.')
