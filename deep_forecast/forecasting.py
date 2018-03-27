# -*- coding: utf-8 -*-
from keras.optimizers import RMSprop
from multivariate_rnns import TruncatedRNN


class RNNRegressor():

    def __init__(self, topology, batch_size=10, n_epochs=1, lr=0.01):

        self._topology = topology
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._learning_rate = lr

        self._model = None
        self._loss = 'mse'
        self._optimizer = RMSprop(lr=lr)
        self._metrics = ['mape']
        self._history = None

        self.is_fitted = False

    def fit(self, X, y):

        self.check_data_format(X, y)

        self._model = TruncatedRNN(
            input_dim=X.shape[2],
            input_length=X.shape[1],
            topology=self._topology
        )

        self._model.compile(
            loss=self._loss,
            optimizer=self._optimizer,
            metrics=self._metrics
        )

        print('Training in progress..')
        self._history = self._model.fit(
            X,
            y,
            batch_size=self._batch_size,
            nb_epochs=self._n_epochs,
            verbose=0,
            validation_split=0.0
        )

        print('Training completed.')
        self.isfitted = True

    def criticise(self):
        self.check_is_fitted()
        return self._history

    def predict(self, X):

        self.check_is_fitted()
        self.check_data_format(X, None)

        return self._model.predict(X)

    @property
    def optimizer(self):
        return self._optimizer.__name__

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @staticmethod
    def check_data_format(X, y):
        pass

    def check_is_fitted(self):
        if not self.is_fitted:
            raise Exception('Model has not been fitted!')
