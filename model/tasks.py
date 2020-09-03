"""This module contains various experimental environments used for testing
human behavior."""
from __future__ import division
from numpy import zeros, ones
from numpy.random import choice, rand


class RevLearn(object):
    """Here we define the environment for the probabilistic reversal learning task
    """

    def __init__(self, observation_generator,
                 state_transition_generator,
                 state_duration_generator,
                 blocks=1, T=10):

        self.T = T
        self.blocks = blocks
        # set probability distribution used for generating observations
        self.ObsGen = observation_generator

        # set probability distribution used for generating state transitions
        self.StGen = state_transition_generator
        self.ns = self.StGen.shape[0]

        # set probability distribution used for generating duration transitions
        self.DurGen = state_duration_generator
        self.nd = self.DurGen.shape[0]

        # set container that keeps track the evolution of the hidden states
        self.hidden_states = zeros((blocks, T, 2), dtype=int)

    def set_hidden_states(self, hidden_states=None):
        if hidden_states is not None:
            self.hidden_states[:] = hidden_states
        else:
            for b in range(self.blocks):
                [s, d] = [choice(self.ns, p=ones(self.ns)/self.ns),
                          choice(self.nd, p=self.DurGen)]

                # evolve hidden states starting from initial state
                for t in range(self.T):
                    self.hidden_states[b, t, :] = [s, d]
                    if d > 0:
                        d = d-1
                    else:
                        [s, d] = [choice(self.ns, p=self.StGen[:, s]),
                                  choice(self.nd, p=self.DurGen)]

        self.observation = zeros(self.hidden_states.shape[:2], dtype=int)

    def generate_observations(self, t, responses):
        # return the observation for given choice
        s = self.hidden_states[:, t, 0]
        rp = self.O[s, responses]  # reward probability
        self.observation[:, t] = (rand(self.blocks) < rp).astype(int)

        return self.observation[:, t]
