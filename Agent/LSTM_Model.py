from keras.layers import Dropout, GRU
from keras import layers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU


def LSTM_Model(input_tensor, gru_cells):
    model_input = input_tensor
    x = GRU(gru_cells, return_sequences=True, recurrent_dropout=0.5)(model_input)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    y = GRU(gru_cells, return_sequences=True, recurrent_dropout=0.5)(x)
    y = BatchNormalization()(y)
    y = PReLU()(y)

    z = layers.add([x, y])
    z = GRU(gru_cells, recurrent_dropout=0.5)(z)
    z = BatchNormalization()(z)
    z = PReLU()(z)

    return z

