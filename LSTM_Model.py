from keras.layers import LSTM
from keras import layers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU


def LSTM_Model(input_tensor, gru_cells):
    model_input = input_tensor
    x = LSTM(gru_cells, return_sequences=True)(model_input)
    x = PReLU()(x)

    y = LSTM(gru_cells, return_sequences=True)(x)
    y = PReLU()(y)

    z = layers.concatenate([x, y])
    z = BatchNormalization()(z)
    z = LSTM(gru_cells)(z)
    z = PReLU()(z)
    z = BatchNormalization()(z)

    return z

