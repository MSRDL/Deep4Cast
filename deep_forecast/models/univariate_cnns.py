# # -*- coding: utf-8 -*-
# from keras.layers import Input, Dense, Conv1D, MaxPooling1D, AveragePooling1D, Dropout, Flatten, GRU
# from keras.layers.advanced_activations import ELU
# from keras.models import Model, Sequential


# class UnivariateCNN(Model):
#     """Extend Model object by including 1d convolutions."""

#     def __init__(self, input_dim, input_length, topology):
#         # Initialize additional properties
#         self.name = 'TemporalCNN'
#         self.input_dim = input_dim
#         self.input_length = input_length
#         self.topology = topology

#         self.__activation = ELU()
#         self.__rnd_init = 'glorot_normal'

#         # Check if topology is valid
#         if len(topology) <  1:
#             pass

#         # Initialize super class with custom layers
#         super(TemporalCNN, self).__init__(*self.__build_layers)

#     def __build_layers(self):
#         input_layer = Input(shape=(self.input_length, self.input_dim))
#         x = Conv1D(64, 5, activation=activation, init=self.__rnd_init)(
#             input_layer
#         )
#         # dim: 16 x 64
#         x = MaxPooling1D(3, stride=1)(x)

#         # x = AveragePooling1D(pool_length=2, stride=1)(x)
#         # dim: 14 x 64
#         x = Conv1D(64, 3, activation=activation, init=self.__rnd_init)(x)
#         # dim: 12 x 64
#         x = MaxPooling1D(4, stride=2)(x)  # global max pooling
#         # x = AveragePooling1D(pool_length=4, stride=1)(x)
#         # dim: 5 x 64

#         x = Conv1D(128, 3, activation=activation, init=self.__rnd_init)(x)
#         # dim: 3 x 128
#         x = MaxPooling1D(3, stride=1)(x)  # global max pooling
#         # x = AveragePooling1D(pool_length=4, stride=1)(x)
#         # dim: 1 x 128

#         x = Flatten()(x)
#         x = Dense(128, activation=activation, init=self.__rnd_init)(x)
#         x = Dropout(0.5)(x)

#         prediction_layer = Dense(self.input_dim)(x)

#         return input_layer, prediction_layer