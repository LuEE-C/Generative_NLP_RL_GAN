# Generative_NLP_RL_GAN
Trying to train a NLP generative model in a reinforcement learning setting.

I currently am trying to train a https://arxiv.org/abs/1710.02298 Rainbow DQN (only Noisy network, C51 and prioritized experience replay, paper seemed to show it gives the biggest gains by far) to generate from a truncated Google Billion Word dataset. 

The general idea is to beat several different environnement, that get progressively harder with my DQN. In this case, the environnement is a discriminator that is trained to differentiate between my DQN's output and the dataset untill it's loss reaches a treshold (0.1 in this case), the reward is the output of the discriminator and we consider the environnement beat when the loss of the discriminator would reach another threshold (0.9 in this case).

The model adds 1 word at the time to a word vector of fixed size and then the discriminator evaluates the full vector.

The model does not currently give satisfying outputs when trained.
