"""Data handling module.
This module provides access to a generic generator that can be used with
forecaster 'fit_generator' function.
"""

import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    ''' inspired by https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly#data-generator '''
    'Generates data for Keras'
    def __init__(self,
                 X_data,
                 y_data,
                 batch_size=16,
                 shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.y_data = y_data
        self.X_data = X_data
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X_data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = self.X_data[indexes]
        y = self.y_data[indexes]
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X_data))
        if self.shuffle:
            np.random.shuffle(self.indexes)