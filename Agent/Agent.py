import os

import keras.backend as K
from keras import losses
import numba as nb
import numpy as np
from keras.layers import Input, Dense, Embedding, PReLU, BatchNormalization, Conv1D, UpSampling1D
from keras.models import Model

from Environnement.Environnement import Environnement
from LSTM_Model import LSTM_Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def policy_loss(actual_value, predicted_value, old_prediction):
    advantage = actual_value - predicted_value

    # Maybe some halfbaked normalization would be nice
    # something like advantage = advantage + 0.1 * advantage/(K.std(advantage) + 1e-10)

    # Fullbaked norm seems very unstable
    # advantage /= (K.std(advantage) + 1e-10)
    def loss(y_true, y_pred):
        prob = K.sum(y_pred * y_true, axis=-1)
        old_prob = K.sum(old_prediction * y_true, axis=-1)
        log_prob = K.log(prob + 1e-10)

        r = prob / (old_prob + 1e-10)

        entropy = K.sum(y_pred * K.log(y_pred + 1e-10), axis=-1)
        return -log_prob * K.mean(K.minimum(r * advantage, K.clip(r, min_value=0.8, max_value=1.2) * advantage)) + 0.01 * entropy
    return loss


class Agent:
    def __init__(self, training_epochs=10, cutoff=4, from_save=False, gamma=.9, batch_size=126):
        self.training_epochs = training_epochs
        self.cutoff = cutoff
        self.environnement = Environnement(cutoff=cutoff)
        self.vocab = self.environnement.different_words

        self.training_data = [[], [], [], []]
        self.batch_size = batch_size

        # Bunch of placeholders values
        self.dummy_value = np.zeros((1, 1))
        self.dummy_predictions = np.zeros((1, self.vocab))

        self.labels = np.array([1] * self.batch_size + [0] * self.batch_size)

        self.actor_critic, self.discriminator = self._build_actor_critic_discriminator()

        self.gammas = np.array([gamma ** (i + 1) for i in range(self.cutoff)]).astype(np.float32)

        if from_save is True:
            self.actor_critic.load_weights('actor_critic')
            self.discriminator.load_weights('discriminator')

    def _build_actor_critic_discriminator(self):

        state_input = Input(shape=(self.cutoff,))

        # Used for loss function
        actual_value = Input(shape=(1,))
        predicted_value = Input(shape=(1,))
        old_predictions = Input(shape=(self.vocab,))

        embedding = Embedding(self.vocab + 1, 50, input_length=self.cutoff)(state_input)

        main_network = Conv1D(256, 3, padding='same')(embedding)
        main_network = PReLU()(main_network)
        main_network = BatchNormalization()(main_network)
        main_network = UpSampling1D()(main_network)

        main_network = Conv1D(128, 5, padding='same')(main_network)
        main_network = PReLU()(main_network)
        main_network = BatchNormalization()(main_network)
        main_network = UpSampling1D()(main_network)

        main_network = Conv1D(64, 7, padding='same')(main_network)
        main_network = PReLU()(main_network)
        main_network = BatchNormalization()(main_network)

        main_network = LSTM_Model(main_network, 75)
        actor_critic = Dense(128)(main_network)
        actor_critic = PReLU()(actor_critic)
        actor_critic = BatchNormalization()(actor_critic)

        discriminator = Dense(128)(main_network)
        discriminator = PReLU()(discriminator)
        discriminator = BatchNormalization()(discriminator)

        actor_next_word = Dense(self.vocab, activation='softmax')(actor_critic)
        critic_value = Dense(1)(actor_critic)
        discriminator_verdict = Dense(1, activation='sigmoid')(discriminator)

        actor_critic = Model(inputs=[state_input, actual_value, predicted_value, old_predictions], outputs=[actor_next_word, critic_value])
        actor_critic.compile(optimizer='adam',
                      loss=[policy_loss(actual_value=actual_value,
                                        predicted_value=predicted_value,
                                        old_prediction=old_predictions
                                        ),
                            'mse'
                            ])

        discriminator = Model(inputs=[state_input],
                             outputs=[discriminator_verdict])
        discriminator.compile(optimizer='adam',
                             loss=['binary_crossentropy'])

        return actor_critic, discriminator

    def train(self, epoch):

        value_list, discrim_losses, policy_losses, critic_losses = [], [], [], []
        e = 0
        while e <= epoch:
            done = False
            print('Epoch :', e)
            batch_num = 0
            while done == False:

                fake_batch, actions, predicted_values, old_predictions = self.get_fake_batch()
                values = self.get_values(fake_batch)
                fake_batch = fake_batch[:self.batch_size]
                value_list.append(np.mean(values))


                tmp_loss = np.zeros(shape=(self.training_epochs, 2))
                for i in range(self.training_epochs):
                    tmp_loss[i] = (self.actor_critic.train_on_batch([fake_batch, values, predicted_values, old_predictions],
                                                                    [actions, values])[1:])
                policy_losses.append(np.mean(tmp_loss[:,0]))
                critic_losses.append(np.mean(tmp_loss[:,1]))

                real_batch, done = self.environnement.query_state(self.batch_size)
                batch = np.vstack((real_batch, fake_batch))
                discrim_losses.append(self.discriminator.train_on_batch([batch], [self.labels])[-1])


                if batch_num % 500 == 0:
                    print()
                    self.actor_critic.save_weights('actor_critic')
                    self.discriminator.save_weights('discriminator')
                    print('Batch number :', batch_num, '\tEpoch :', e, '\tAverage values :', np.mean(value_list))
                    print('Policy losses :', '%.5f' % np.mean(policy_losses),
                          '\tCritic losses :', '%.5f' % np.mean(critic_losses),
                          '\tDiscriminator losses :', '%.5f' % np.mean(discrim_losses))
                    self.print_pred()
                    value_list, discrim_losses, policy_losses, critic_losses = [], [], [], []

                batch_num += 1
            e += 1

    @nb.jit
    def make_seed(self):


        # This is the kinda Z vector
        seed = np.random.random_integers(low=0, high=self.vocab - 1, size=(1, self.cutoff))

        predictions = self.actor_critic.predict(
            [seed, self.dummy_value, self.dummy_value, self.dummy_predictions])[0]
        for _ in range(self.cutoff - 1):
            numba_optimised_seed_switch(predictions[0], self.vocab, seed)
            predictions = self.actor_critic.predict(
                [seed, self.dummy_value, self.dummy_value, self.dummy_predictions])[0]
        numba_optimised_seed_switch(predictions[0], self.vocab, seed)

        return seed

    @nb.jit
    def get_fake_batch(self):

        seed = self.make_seed()
        fake_batch = np.zeros((self.batch_size + self.cutoff, self.cutoff))
        predicted_values = np.zeros((self.batch_size + self.cutoff, 1))
        actions = np.zeros((self.batch_size + self.cutoff, self.vocab))
        old_predictions = np.zeros((self.batch_size + self.cutoff, self.vocab))
        for i in range(self.batch_size + self.cutoff):
            predictions = self.actor_critic.predict([seed, self.dummy_value, self.dummy_value, self.dummy_predictions])
            numba_optimised_pred_rollover(old_predictions, predictions, self.vocab, i, seed, predicted_values, actions, fake_batch)

        return fake_batch, actions[:self.batch_size], predicted_values[:self.batch_size], old_predictions[:self.batch_size]

    @nb.jit
    def get_values(self, fake_batch):
        # Subtract 0.5 for better understanding of values
        values = self.discriminator.predict([fake_batch])[-1] - .5

        return numba_optimised_nstep_value_function(values, self.batch_size, self.cutoff, self.gammas)

    @nb.jit
    def print_pred(self):
        fake_state = self.make_seed()

        pred = ""
        for j in range(self.cutoff):
            pred += self.environnement.ind_to_word[fake_state[0][j]]
            pred += " "
        print(pred)

        fake_state = self.make_seed()
        pred = ""
        for j in range(self.cutoff):
            pred += self.environnement.ind_to_word[fake_state[0][j]]
            pred += " "
        print(pred)


# Some strong numba optimisation in bottlenecks
# N_Step reward function
@nb.jit(nb.float32[:,:](nb.float32[:,:], nb.int64, nb.int64, nb.float32[:]))
def numba_optimised_nstep_value_function(values, batch_size, cutoff, gammas):
    for i in range(batch_size):
        for j in range(cutoff):
            values[i] += values[i + j + 1] * gammas[j]
    return values[:batch_size]


@nb.jit(nb.void(nb.float32[:,:], nb.float32[:,:], nb.int64, nb.int64, nb.float32[:,:], nb.float32[:,:], nb.float32[:,:], nb.float32[:,:]))
def numba_optimised_pred_rollover(old_predictions, predictions, vocab, index, seed, predicted_values, actions, fake_batch):
    old_predictions[index] = predictions[0][0]

    choice = np.random.choice(vocab, 1, p=predictions[0][0])
    predicted_values[index][0] = predictions[1][0]
    actions[index][choice] = 1
    seed[:, :-1] = seed[:, 1:]
    seed[:, -1] = choice
    fake_batch[index] = seed

@nb.jit(nb.void(nb.float32[:,:], nb.int64, nb.float32[:,:]))
def numba_optimised_seed_switch(predictions, vocab, seed):
    seed[:, :-1] = seed[:, 1:]
    seed[:, -1] = np.random.choice(vocab, 1, p=predictions)


if __name__ == '__main__':
    agent = Agent(cutoff=6, from_save=False, batch_size=128)
    agent.train(epoch=5000)