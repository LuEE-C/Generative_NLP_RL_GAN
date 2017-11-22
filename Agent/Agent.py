import os

import numba as nb
import numpy as np
import math
from random import random, randint

from keras.optimizers import Adam
from keras.layers import Input, Dense, Embedding, PReLU, BatchNormalization, Conv1D
from keras.models import Model

from Environnement.Environnement import Environnement
from LSTM_Model import LSTM_Model
from NoisyDense import NoisyDense
from PriorityExperienceReplay.PriorityExperienceReplay import Experience

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Agent:
    def __init__(self, cutoff=8, from_save=False, gamma=.9, batch_size=32, min_history=64000, lr=0.0000625,
                 sigma_init=0.5, target_network_period=32000, adam_e=1.5*10e-4, atoms=51,
                 discriminator_loss_limits=0.1, n_steps=3):

        self.cutoff = cutoff
        self.environnement = Environnement(cutoff=cutoff, min_frequency_words=300000)
        self.vocab = self.environnement.different_words

        self.batch_size = batch_size

        self.n_steps = n_steps

        self.labels = np.array([1] * self.batch_size + [0] * self.batch_size)
        self.gammas = np.array([gamma ** (i + 1) for i in range(self.n_steps + 1)]).astype(np.float32)

        self.atoms = atoms
        self.v_max = np.sum([0.5 * gam for gam in self.gammas])
        self.v_min = - self.v_max
        self.delta_z = (self.v_max - self.v_min) / float(self.atoms - 1)
        self.z_steps = np.array([self.v_min + i * self.delta_z for i in range(self.atoms)]).astype(np.float32)

        self.epsilon_greedy_max = 0.8
        self.sigma_init = sigma_init


        self.min_history = min_history
        self.lr = lr
        self.target_network_period = target_network_period
        self.adam_e = adam_e

        self.discriminator_loss_limit = discriminator_loss_limits

        self.model, self.target_model = self._build_model(), self._build_model()
        self.discriminator = self._build_discriminator()

        self.dataset_epoch = 0
        if from_save is True:
            self.model.load_weights('model')
            self.target_model.load_weights('model')
            self.discriminator.load_weights('discriminator')

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_average_noisy_weight(self):
        average = []
        for i in range(self.vocab):
            average.append(np.mean(self.model.get_layer('Word_'+str(i)).get_weights()[1]))

        return np.mean(average), np.std(average)

    def _build_model(self):

        state_input = Input(shape=(self.cutoff,))

        embedding = Embedding(self.vocab + 1, 50, input_length=self.cutoff)(state_input)

        main_network = Conv1D(256, 3, padding='same')(embedding)
        main_network = PReLU()(main_network)

        main_network = LSTM_Model(main_network, 100, batch_norm=False)

        main_network = Dense(256)(main_network)
        main_network = PReLU()(main_network)

        main_network = Dense(512)(main_network)
        main_network = PReLU()(main_network)

        dist_list = []

        for i in range(self.vocab):
            dist_list.append(NoisyDense(self.atoms, activation='softmax', sigma_init=self.sigma_init, name='Word_' + str(i))(main_network))


        actor = Model(inputs=[state_input], outputs=dist_list)
        actor.compile(optimizer=Adam(lr=self.lr, epsilon=self.adam_e),
                      loss='categorical_crossentropy')

        return actor

    def _build_discriminator(self):

        state_input = Input(shape=(self.cutoff,))

        embedding = Embedding(self.vocab + 1, 50, input_length=self.cutoff)(state_input)

        main_network = Conv1D(256, 3, padding='same')(embedding)
        main_network = PReLU()(main_network)
        main_network = BatchNormalization()(main_network)

        main_network = LSTM_Model(main_network, 100)

        main_network = Dense(256)(main_network)
        main_network = PReLU()(main_network)
        main_network = BatchNormalization()(main_network)

        main_network = Dense(512)(main_network)
        main_network = PReLU()(main_network)
        main_network = BatchNormalization()(main_network)

        discriminator_output = Dense(1, activation='sigmoid')(main_network)


        discriminator = Model(inputs=[state_input], outputs=discriminator_output)
        discriminator.compile(optimizer=Adam(),
                      loss='binary_crossentropy')

        discriminator.summary()

        return discriminator

    def train(self, epoch):

        e, total_frames = 0, 0
        while e <= epoch:
            print('Epoch :', e)

            discrim_loss, model_loss_array, memory = [1], [], Experience(memory_size=1000000, batch_size=self.batch_size, alpha=0.5)
            while np.mean(discrim_loss[-20:]) >= self.discriminator_loss_limit:
                discrim_loss.append(self.train_discriminator())

            for i in range(self.min_history//200):
                states, rewards, actions, states_prime = self.get_training_batch(200, self.get_epsilon(np.mean(discrim_loss[-20:])))
                for j in range(200):
                    memory.add((states[j], rewards[j], actions[j], states_prime[j]), 5)


            trained_frames = 1
            while np.mean(discrim_loss[-20:]) < 0.5 + 0.5 * 500000/(trained_frames * 10 * 4 * self.batch_size):

                if trained_frames % (self.target_network_period//(10 * 4 * self.batch_size)) == 0:
                    self.update_target_model()

                states, rewards, actions, states_prime = self.get_training_batch(10 * self.batch_size, self.get_epsilon(np.mean(discrim_loss[-20:])))
                for j in range(10 * self.batch_size):
                    memory.add((states[j], rewards[j], actions[j], states_prime[j]), 5)
                for j in range(10 * 4):
                    out, weights, indices = memory.select(min(1, 0.4 + 1.2 * np.mean(discrim_loss[-20:]))) # Scales b value
                    model_loss_array.append(self.train_on_replay(out, self.batch_size)[0])
                    memory.priority_update(indices, [model_loss_array[-1] for _ in range(self.batch_size)])

                trained_frames += 1
                total_frames += 1
                discrim_loss.append(self.train_discriminator(evaluate=True))

                if trained_frames % 100 == 0:
                    print()
                    mean, std = self.get_average_noisy_weight()
                    print('Average loss of model :', np.mean(model_loss_array[-10 * 4 * 20:]),
                          '\tAverage discriminator loss :', np.mean(discrim_loss[-20:]),
                          '\tFrames passed :', trained_frames * 10 * 4 * self.batch_size,
                          '\tTotal frames passed :', total_frames * 10 * 4 * self.batch_size,
                          '\tAverage Noisy Weights :', mean,
                          '\tSTD Noisy Weights :', std,
                          '\tEpoch :', e,
                          '\tDataset Epoch :', self.dataset_epoch
                          )

                    self.print_pred()
                    self.print_pred()

            self.update_target_model()

            e += 1

    def get_epsilon(self, discrim_loss):
        epsilon = min(1.0, (0.1 / discrim_loss)) * self.epsilon_greedy_max
        return epsilon

    @nb.jit
    def train_discriminator(self, evaluate=False):
        fake_batch = self.get_fake_batch()
        real_batch, done = self.environnement.query_state(self.batch_size)
        if done is True:
            self.dataset_epoch += 1
            print('Current Dataset Epoch :', self.dataset_epoch)
        batch = np.vstack((real_batch, fake_batch))
        if evaluate is True:
            return self.discriminator.evaluate([batch], [self.labels], verbose=0)
        return self.discriminator.train_on_batch([batch], [self.labels])

    @nb.jit
    def make_seed(self, seed=None):
        if seed is None:
            # This is the kinda Z vector
            seed = np.random.random_integers(low=0, high=self.vocab - 1, size=(1, self.cutoff))

        predictions = self.target_model.predict(seed)
        for _ in range(self.cutoff - 1):
            numba_optimised_seed_switch(predictions, seed, self.z_steps)
            predictions = self.target_model.predict(seed)
        numba_optimised_seed_switch(predictions, seed, self.z_steps)

        return seed

    @nb.jit
    def get_fake_batch(self):

        seed = self.make_seed()
        fake_batch = np.zeros((self.batch_size, self.cutoff))
        for i in range(self.batch_size):
            predictions = self.target_model.predict([seed])
            numba_optimised_pred_rollover(predictions, i, seed, fake_batch, self.z_steps)

        return fake_batch

    @nb.jit
    def get_training_batch(self, batch_size, epsilon):
        seed = self.make_seed()
        states = np.zeros((batch_size + self.n_steps, self.cutoff))
        actions = np.zeros((batch_size + self.n_steps, 1))
        for i in range(batch_size + self.n_steps):
            action = -1
            predictions = self.target_model.predict(seed)
            if random() < epsilon:
                action = randint(0, self.vocab - 1)

            numba_optimised_pred_rollover_with_actions(predictions, i, seed, states, self.z_steps, actions, action)

        rewards = self.get_values(states)
        states_prime = states[self.n_steps:]

        return states[:-self.n_steps], rewards, actions, states_prime

    @nb.jit
    def get_values(self, fake_batch):
        values = self.discriminator.predict(fake_batch)
        return numba_optimised_nstep_value_function(values, values.shape[0], self.n_steps, self.gammas)

    @nb.jit
    def print_pred(self):
        fake_state = self.make_seed()

        pred = ""
        for _ in range(4):
            for j in range(self.cutoff):
                pred += self.environnement.ind_to_word[fake_state[0][j]]
                pred += " "
            fake_state = self.make_seed(fake_state)
        for j in range(self.cutoff):
            pred += self.environnement.ind_to_word[fake_state[0][j]]
            pred += " "
        print(pred)



    # @nb.jit
    def train_on_replay(self, data, batch_size):
        states, reward, actions, state_prime = make_dataset(data=data, batch_size=batch_size)

        m_prob = np.zeros((batch_size, self.vocab, self.atoms))

        z = self.target_model.predict(state_prime)
        z = np.array(z)
        z = np.swapaxes(z, 0, 1)
        q = np.sum(np.multiply(z, self.z_steps), axis=-1)
        optimal_action_idxs = np.argmax(q, axis=-1)

        update_m_prob(self.batch_size, self.atoms, self.v_max, self.v_min, reward, self.gammas[-1],
                      self.z_steps, self.delta_z, m_prob, actions, z, optimal_action_idxs)

        return self.model.train_on_batch(states, [m_prob[:,i,:] for i in range(self.vocab)])


@nb.jit(nb.void(nb.int64,nb.int64,nb.float32,nb.float32, nb.float32[:],nb.float32,
                nb.float32[:],nb.float32,nb.float32[:,:,:],nb.float32[:,:], nb.float32[:,:,:], nb.float32[:]))
def update_m_prob(batch_size, atoms, v_max, v_min, reward, gamma, z_steps, delta_z, m_prob, actions, z, optimal_action_idxs):
    for i in range(batch_size):
        for j in range(atoms):
            Tz = min(v_max, max(v_min, reward[i] + gamma * z_steps[j]))
            bj = (Tz - v_min) / delta_z
            m_l, m_u = math.floor(bj), math.ceil(bj)
            m_prob[i, actions[i, 0], int(m_l)] += z[i, optimal_action_idxs[i], j] * (m_u - bj)
            m_prob[i, actions[i, 0], int(m_l)] += z[i, optimal_action_idxs[i], j] * (bj - m_l)

# @nb.jit
def make_dataset(data, batch_size):
    states, reward, actions, state_prime = [], [], [], []
    for i in range(batch_size):
        states.append(data[i][0])
        reward.append(data[i][1])
        actions.append(data[i][2])
        state_prime.append(data[i][3])
    states = np.array(states)
    reward = np.array(reward)
    actions = np.array(actions).astype(np.int)
    state_prime = np.array(state_prime)
    return states, reward, actions, state_prime

@nb.jit(nb.int64(nb.float32[:,:], nb.float32[:]))
def get_optimal_action(z, z_distrib):

    z_concat = np.vstack(z)
    q = np.sum(np.multiply(z_concat, z_distrib), axis=1)
    action_idx = np.argmax(q)

    return action_idx


# Some strong numba optimisation in bottlenecks
# N_Step reward function
@nb.jit(nb.float32[:,:](nb.float32[:,:], nb.int64, nb.int64, nb.float32[:]))
def numba_optimised_nstep_value_function(values, batch_size, n_step, gammas):
    for i in range(batch_size):
        for j in range(n_step):
            values[i] += values[i + j + 1] * gammas[j]
    return values[:batch_size]


@nb.jit(nb.void(nb.float32[:,:], nb.int64, nb.float32[:,:], nb.float32[:,:], nb.float32[:]))
def numba_optimised_pred_rollover(predictions, index, seed, fake_batch, z_distrib):
    seed[:, :-1] = seed[:, 1:]
    seed[:, -1] = get_optimal_action(predictions, z_distrib)
    fake_batch[index] = seed

@nb.jit(nb.void(nb.float32[:,:], nb.int64, nb.float32[:,:], nb.float32[:,:], nb.float32[:], nb.float32[:,:], nb.int64))
def numba_optimised_pred_rollover_with_actions(predictions, index, seed, fake_batch, z_distrib, actions, action):
    if action != -1:
        choice = action
    else:
        choice = get_optimal_action(predictions, z_distrib)
    seed[:, :-1] = seed[:, 1:]
    seed[:, -1] = choice
    actions[index] = choice
    fake_batch[index] = seed

@nb.jit(nb.void(nb.float32[:,:], nb.int64, nb.float32[:,:]))
def numba_optimised_seed_switch(predictions, seed, z_distrib):
    seed[:, :-1] = seed[:, 1:]
    seed[:, -1] = get_optimal_action(predictions, z_distrib)


if __name__ == '__main__':
    agent = Agent(cutoff=5, from_save=False, batch_size=32)
    agent.train(epoch=5000)
