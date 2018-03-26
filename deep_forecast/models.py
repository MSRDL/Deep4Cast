
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, AveragePooling1D, Dropout, Flatten, GRU
from keras.layers.advanced_activations import ELU
from keras.models import Model, Sequential


class TemporalCNN(Model):
    """Extend Model object by including 1d convolutions."""

    def __init__(self, input_dim, input_length, topology):
        # Initialize additional properties
        self.name = 'TemporalCNN'
        self.input_dim = input_dim
        self.input_length = input_length
        self.topology = topology

        self.__activation = ELU()
        self.__rnd_init = 'glorot_normal'

        # Check if topology is valid
        if len(topology) <  

        # Initialize super class with custom layers
        super(TemporalCNN, self).__init__(*self.__build_layers)

    def __build_layers(self):
        input_layer = Input(shape=(self.input_length, self.input_dim))
        x = Conv1D(64, 5, activation=activation, init=self.__rnd_init)(
            input_layer
        )
        # dim: 16 x 64
        x = MaxPooling1D(3, stride=1)(x)

        # x = AveragePooling1D(pool_length=2, stride=1)(x)
        # dim: 14 x 64
        x = Conv1D(64, 3, activation=activation, init=self.__rnd_init)(x)
        # dim: 12 x 64
        x = MaxPooling1D(4, stride=2)(x)  # global max pooling
        # x = AveragePooling1D(pool_length=4, stride=1)(x)
        # dim: 5 x 64

        x = Conv1D(128, 3, activation=activation, init=self.__rnd_init)(x)
        # dim: 3 x 128
        x = MaxPooling1D(3, stride=1)(x)  # global max pooling
        # x = AveragePooling1D(pool_length=4, stride=1)(x)
        # dim: 1 x 128

        x = Flatten()(x)
        x = Dense(128, activation=activation, init=self.__rnd_init)(x)
        x = Dropout(0.5)(x)

        prediction_layer = Dense(self.input_dim)(x)

        return input_layer, prediction_layer


class VannillaRNN(Sequential):
    """Extend Sequential model object by including vanilla layers."""

    def __init__(self, input_dim, input_length, topology, unit=GRU):
        # Initialize additional properties
        self.name = 'VanillaRNNwith' + unit.__name__
        self.input_dim = input_dim
        self.input_length = input_length
        self.topology = topology

        self.__unit = unit
        self.__rnd_init = 'glorot_normal'

        # Initialize super class with custom layers
        super(VannillaRNN, self).__init__(*self.__build_layers)

    def __build_layers(self):
        """Build RNN model layers."""

        # Add RNN layers one at a time with layerwise output for easy
        # inspection of internal layers after model is trained
        n_layers = len(topology)
        layers = []
        for i in range(n_layers):
            return_sequences = True if i < n_layers - 1 else False
            layer = self.__unit(
                self.n_rnn_units,
                input_length=self.input_length,
                input_dim=self.input_dim,
                return_sequences=return_sequences
            )
            layers.append(layer)

        # Add dense output later
        layers.append(Dense(self.input_dim, init=self.__rnd_init))

        return layers
