from Environnement.Environnement import Environnement
from Agent.NoisyDense import NoisyDense
from Agent.LSTM_Model import LSTM_Model
from Agent.DenseNet import DenseNet

import os
import pickle
import numpy as np
import numba
import math

import keras.backend as K
from keras.layers import Input, Dense, Embedding
from keras.models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def normal_log_density(x, policy):
    var = K.pow(K.std(policy), 2)
    log_density = -K.pow(x - K.mean(policy), 2) / (2 * var) - 0.5 * K.log(2 * math.pi) - K.log(K.std(policy))
    return K.sum(log_density)

def proximal_policy_optimization_loss(actual_value,  old_prediction):
    # Fucky advantage, really just reward but it will be self correcting like GAN's are hopefully
    advantage = actual_value
    def loss(y_true, y_pred):
        prob = normal_log_density(y_pred, y_true)
        old_prob = normal_log_density(y_pred, old_prediction)

        r = prob/old_prob

        return K.mean(K.minimum(r * advantage, K.clip(r, min_value=0.8, max_value=1.2) * advantage))
    return loss


class Agent:
    def __init__(self, training_epochs=10, sigma_init=0.02, LSTM_n=100, cutoff=4):
        self.training_epochs = training_epochs
        self.cutoff = cutoff
        self.environnement = Environnement(cutoff=cutoff)
        self.vocab = self.environnement.different_words

        self.training_data = [[], [], [], []]
        self.dummy_value = np.zeros((1, 1))
        self.dummy_prediction = np.zeros((1, self.vocab))

        self.values, self.amount_of_trades, self.val_values, self.val_amount_of_trades = [], [], [], []

        self.actor = self._build_actor(sigma_init=sigma_init, LSTM_n=LSTM_n)
        self.critic = self._build_critic(LSTM_n=LSTM_n)

    def _build_actor(self, sigma_init, LSTM_n):

        state_input = Input(shape=(self.cutoff))
        actual_value = Input(shape=(1,))
        old_prediction = Input(shape=(self.vocab,))
        x = Embedding(self.vocab, 60, input_length=self.cutoff)(state_input)
        #x = DenseNet(state_input, 5, 4, 16, 20, dropout_rate=0.5)
        x = LSTM_Model(x, LSTM_n)

        next_word = NoisyDense(self.vocab, activation='softmax', sigma_init=sigma_init, name='next_word')(x)

        model = Model(inputs=[state_input, actual_value, old_prediction], outputs=[next_word])
        model.compile(optimizer='adam',
                      loss=[proximal_policy_optimization_loss(
                          actual_value=actual_value,
                          old_prediction=old_prediction)])

        model.get_layer('next_word').sample_noise()
        model.summary()
        return model

    def _build_critic(self, LSTM_n):

        state_input = Input(shape=(self.cutoff,))
        x = Embedding(self.vocab, 60, input_length=self.cutoff)(state_input)
        #x = DenseNet(state_input, 5, 4, 16, 20, dropout_rate=0.5)
        x = LSTM_Model(x, LSTM_n)

        out_value = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer='adam', loss=['binary_crossentropy'])
        return model

    def train(self, epoch, batch_size=32):

        #fixing batch_size so we don't have odd data and such, iunno if it will change anything but it can't hurt
        if batch_size % self.cutoff != 0:
            batch_size += (self.cutoff - batch_size % self.cutoff)

        e = -1
        while e <= epoch:
            done = False
            e += 1
            print('Epoch :', e)
            while done == False:

                real_batch, done = self.environnement.query_state(batch_size=batch_size)
                fake_batch = self.get_fake_batch(batch_size=batch_size)

                batch = np.append(real_batch, fake_batch)
                labels = np.append(np.ones((batch_size,)),np.zeros((batch_size,)))

                #todo value calculation to train the actor, this almost works.

                # for _ in range(self.training_epochs):
                #     self.actor.train_on_batch([real_state, v, value_predictions, old_prediction], [a])
                # for _ in range(self.training_epochs):
                #     self.critic.train_on_batch([real_state], [v])
                self.actor.get_layer('next_word').sample_noise()

    @numba.jit
    def get_fake_batch(self, batch_size):

        fake_batch = np.zeros((batch_size, self.cutoff))
        for i in range(batch_size//self.cutoff):
            fake_state = fake_batch[i]
            for j in range(self.cutoff):
                fake_pred = self.actor.predict([fake_state, self.dummy_value, self.dummy_prediction])
                fake_state[j] = np.argmax(fake_pred)
                fake_batch[i + j] = fake_state
        return fake_batch







if __name__ == '__main__':

    agent = Agent(LSTM_n=50, cutoff=4)
    agent.train(epoch=5000, batch_size=64)