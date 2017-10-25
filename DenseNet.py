# Mostly from https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/DenseNet/densenet.py

from keras.layers.core import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import AveragePooling1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2


def conv_factory(x, nb_filter, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 3Conv1D, optional dropout
    :param x: Input keras network
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras network with b_norm, relu and convolution2d added
    :rtype: keras network
    """

    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = PReLU()(x)
    x = Conv1D(nb_filter, 3,
               kernel_initializer="he_uniform",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition(x, nb_filter, dropout_rate=None, weight_decay=1E-4, compression_rate=0.5):
    """Apply BatchNorm, Relu 1Conv1D, optional dropout and Maxpooling1D
    :param x: keras model
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: model
    :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool
    """
    nb_filter = int(nb_filter*compression_rate)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = PReLU()(x)
    x = Conv1D(nb_filter, 1,
               kernel_initializer="he_uniform",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling1D()(x)

    return x, nb_filter


def denseblock(x, nb_layers, nb_filter, growth_rate,
               dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each
       conv_factory is fed to subsequent ones
    :param x: keras model
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model
    """

    list_feat = [x]
    for i in range(nb_layers):
        x = conv_factory(x, growth_rate, dropout_rate, weight_decay)
        list_feat.append(x)
        x = concatenate(list_feat)
        nb_filter += growth_rate

    return x, nb_filter


def DenseNet(input_tensor, nb_layers, nb_dense_block, growth_rate,
             nb_filter, dropout_rate=None, weight_decay=1E-4, compression_rate=0.5):
    """ Build the DenseNet model
    :param input_tensor: keras functionnal api tensor
    :param nb_layers: int -- how many layers in 1 dense block
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay
    :param compression_rate: float -- compression_rate
    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model
    """

    model_input = input_tensor

    # Initial convolution
    x = Conv1D(nb_filter, 7,
               kernel_initializer="he_uniform",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(model_input)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = denseblock(x, nb_layers, nb_filter, growth_rate,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
        # add transition
        x, nb_filter = transition(x, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay, compression_rate=compression_rate)

    # The last denseblock does not have a transition
    x, nb_filter = denseblock(x, nb_layers, nb_filter, growth_rate,
                              dropout_rate=dropout_rate,
                              weight_decay=weight_decay)

    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = PReLU()(x)
    # Skiped Global average pooling, more interesting to feed found patterns to lstm
    x = GlobalAveragePooling1D()(x)

    return x