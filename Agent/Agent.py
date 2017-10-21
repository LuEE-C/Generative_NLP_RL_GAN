import math
import os

import keras.backend as K
import numba
import numpy as np
from keras.layers import Input, Dense, Embedding, Dropout, BatchNormalization, PReLU
from keras.models import Model

from Environnement.Environnement import Environnement
from LSTM_Model import LSTM_Model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def policy_loss(actual_value, predicted_value, old_prediction):
    advantage = actual_value - predicted_value
    def loss(y_true, y_pred):
        prob = K.sum(y_pred * y_true, axis=1, keepdims=True)
        old_prob = K.sum(old_prediction * y_true, axis=1, keepdims=True)
        log_prob = K.log(prob + 1e-10)

        r = prob / old_prob

        entropy = K.sum(y_pred * K.log(y_pred + 1e-10), axis=1, keepdims=True)
        return -log_prob * K.mean(K.minimum(r * advantage, K.clip(r, min_value=0.8, max_value=1.2) * advantage)) + 1 * entropy
    return loss


class Agent:
    def __init__(self, training_epochs=10, LSTM_n=100, cutoff=4, from_save=False, gamma=.95):
        self.training_epochs = training_epochs
        self.cutoff = cutoff
        self.environnement = Environnement(cutoff=cutoff)
        self.vocab = self.environnement.different_words

        self.training_data = [[], [], [], []]
        self.dummy_value = np.zeros((1, 1))
        self.dummy_predictions = np.zeros((1, self.vocab))

        self.values, self.amount_of_trades, self.val_values, self.val_amount_of_trades = [], [], [], []

        self.actor, self.discriminator = self._build_shared_embedding_actor_critic_and_discriminator(LSTM_n=LSTM_n)

        self.gammas = [gamma ** (i + 1) for i in range(self.cutoff)]

        if from_save is True:
            self.actor.load_weights('actor')
            self.discriminator.load_weights('discriminator')

    def _build_shared_embedding_actor_critic_and_discriminator(self, LSTM_n):

        actor_state_input = Input(shape=(self.cutoff,))
        discriminator_state_input = Input(shape=(self.cutoff,))

        # Used for loss function
        actual_value = Input(shape=(1,))
        predicted_value = Input(shape=(1,))
        old_predictions = Input(shape=(self.vocab,))

        shared_embedding = Embedding(self.vocab + 1, 50, input_length=self.cutoff)

        actor = shared_embedding(actor_state_input)
        discriminator = shared_embedding(discriminator_state_input)

        actor = LSTM_Model(actor, LSTM_n)
        actor = Dense(256, activation='tanh')(actor)
        # ReLu is bad with softmax
        actor = BatchNormalization()(actor)
        discriminator = LSTM_Model(discriminator, LSTM_n)

        actor_next_word = Dense(self.vocab, activation='softmax')(actor)
        critic_value = Dense(1)(actor)
        discriminator_verdict = Dense(1, activation='sigmoid')(discriminator)

        actor_model = Model(inputs=[actor_state_input, actual_value, predicted_value, old_predictions], outputs=[actor_next_word, critic_value])
        actor_model.compile(optimizer='adam',
                      loss=[policy_loss(actual_value=actual_value,
                                        predicted_value=predicted_value,
                                        old_prediction=old_predictions),
                            'mse'])

        discriminator_model = Model(inputs=discriminator_state_input, outputs=discriminator_verdict)
        discriminator_model.compile(optimizer='adam', loss='binary_crossentropy')

        actor_model.summary()

        return actor_model, discriminator_model


    def train(self, epoch, batch_size=32):
        value_list = []
        e = 0
        while e <= epoch:
            done = False
            print('Epoch :', e)
            batch_num = 0
            while done == False:

                fake_batch, actions, predicted_values, old_predictions = self.get_fake_batch(batch_size=batch_size)
                values = self.get_values(batch_size=batch_size, fake_batch=fake_batch)
                fake_batch = fake_batch[:batch_size]
                value_list.append(np.mean(values))

                # print('batch', fake_batch)
                # print('old preds', old_predictions)
                for _ in range(10):
                    self.actor.train_on_batch([fake_batch, values, predicted_values, old_predictions], [actions, values])

                ## Need to penalize critic somehow, maybe implement wassertein afterall?
                ## maybe just train it when gen caught up to it
                # if np.mean(value_list[-100:]) > -0.25:
                real_batch, done = self.environnement.query_state(batch_size=batch_size)
                batch = np.vstack((real_batch, fake_batch))
                labels = np.array([1] * batch_size + [0] * batch_size)
                self.discriminator.train_on_batch([batch], [labels])


                if batch_num % 500 == 0:
                    self.actor.save_weights('actor')
                    self.discriminator.save_weights('discriminator')
                    print('Batch number :', batch_num, '  Epoch :', e, '  Average values :', np.mean(value_list[-100:]))
                    self.print_pred()

                batch_num += 1
            e += 1

    @numba.jit
    def make_seed(self):
        # This is the kinda Z vector
        seed = np.random.random_integers(low=0, high=self.vocab - 1, size=(1, self.cutoff))
        for _ in range(self.cutoff):
            seed[:,:-1] = seed[:,1:]
            predictions = self.actor.predict([seed, self.dummy_value, self.dummy_value, self.dummy_predictions])

            # Numerical stability stuff
            np.nan_to_num(predictions[0][0], copy=False)
            if np.sum(predictions[0][0]) < 0.9:
                print('It broke')
            predictions[0][0] += 1e-10

            # If sum is 1.01 we divide everything by 1.01
            predictions[0][0] /= np.sum(predictions[0][0])

            seed[:,-1] = np.random.choice(self.vocab, 1, p=predictions[0][0])
        return seed

    # If this is to slow I can make multiple batches at the same time
    @numba.jit
    def get_fake_batch(self, batch_size):

        seed = self.make_seed()
        fake_batch = np.zeros((batch_size + self.cutoff, self.cutoff))
        predicted_values = np.zeros((batch_size + self.cutoff, 1))
        actions = np.zeros((batch_size + self.cutoff, self.vocab))
        old_predictions = np.zeros((batch_size + self.cutoff, self.vocab))
        for i in range(batch_size + self.cutoff):
            predictions = self.actor.predict([seed, self.dummy_value, self.dummy_value, self.dummy_predictions])

            np.nan_to_num(predictions[0][0], copy=False)
            predictions[0][0] += 1e-10
            predictions[0][0] /= np.sum(predictions[0][0])

            old_predictions[i] = predictions[0][0]

            choice = np.random.choice(self.vocab, 1, p=predictions[0][0])
            predicted_values[i][0] = predictions[1][0]
            actions[i][choice] = 1
            seed[:,:-1] = seed[:,1:]
            seed[:,-1] = choice
            fake_batch[i] = seed

        return fake_batch, actions[:batch_size], predicted_values[:batch_size], old_predictions[:batch_size]


    # Optimise this 2
    @numba.jit
    def get_values(self, batch_size, fake_batch):
        # Sigmoid output, we 0 center it by substracting 0.5
        values = self.discriminator.predict(fake_batch) - 0.5

        # N_Step reward function
        for i in range(batch_size):
            for j in range(min(self.cutoff, batch_size - i - self.cutoff - 1)):
                values[i] += values[i + j + 1] * self.gammas[j]
        return values[:batch_size]

    @numba.jit
    def print_pred(self):
        fake_state = self.make_seed()
        pred = ""
        for j in range(self.cutoff):
            pred += self.environnement.ind_to_word[fake_state[0][j]]
            pred += " "
        print(pred)

if __name__ == '__main__':
    agent = Agent(LSTM_n=75, cutoff=8, from_save=False)
    # Small batch size makes all the diff, gotta get sum of that stochastic noise
    agent.train(epoch=5000, batch_size=128)