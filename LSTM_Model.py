from keras.layers import CuDNNLSTM
from keras import layers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU


def LSTM_Model(input_tensor, gru_cells, batch_norm=True):
    model_input = input_tensor
    x = CuDNNLSTM(gru_cells, return_sequences=True)(model_input)
    x = PReLU()(x)

    y = CuDNNLSTM(gru_cells, return_sequences=True)(x)
    y = PReLU()(y)

    z = layers.concatenate([x, y])
    if batch_norm is True:
        z = BatchNormalization()(z)
    z = CuDNNLSTM(gru_cells)(z)
    z = PReLU()(z)
    if batch_norm is True:
        z = BatchNormalization()(z)

    return z

