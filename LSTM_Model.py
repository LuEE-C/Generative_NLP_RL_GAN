from keras.layers import Dropout, GRU, Bidirectional, Activation
from keras import layers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU


def LSTM_Model(input_tensor, gru_cells):
    model_input = input_tensor
    x = Bidirectional(GRU(gru_cells, recurrent_dropout=0.5, return_sequences=True))(model_input)
    x = PReLU()(x)

    y = Bidirectional(GRU(gru_cells, recurrent_dropout=0.5, return_sequences=True))(x)
    y = PReLU()(y)

    z = layers.add([x, y])
    z = Bidirectional(GRU(gru_cells, recurrent_dropout=0.5))(z)
    z = PReLU()(z)
    z = BatchNormalization()(z)

    return z

