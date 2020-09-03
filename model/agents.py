"""This module contains the class that defines the interaction between
different modules that govern agent's behavior.
"""
from __future__ import division
import numpy as np


class BayesianReversalLearner(object):

    def __init__(self, transition_matrix,
                 prior_states=None,
                 prior_durations=None,
                 prior_policies=None,
                 tau=1e-10,
                 blocks=1, T=100,
                 number_of_choices=2):

        self.blocks = blocks
        self.T = T

        self.tau = tau  # response noise

        # set parameters of the agent
        self.ns = transition_matrix.shape[0]
        self.nd = transition_matrix.shape[1]

        # set transition matrix
        self.trans_matrix = transition_matrix

        # array containing possible actions
        self.actions = np.arange(self.ns)

        if prior_states is not None:
            sd_prior = prior_states['sd']  # prior beliefs over states and durations
            ab_prior = prior_states['ab']  # prior beliefs over reward probabilities
        else:
            sdp = np.ones((self.ns, self.nd))
            sdp /= sdp.sum()
            sd_prior = np.tile(sdp, self.blocks)
            ab_prior = np.ones((self.blocks, self.ns, 2))

        # set various data structures
        self.desired_actions = np.zeros((blocks, T), dtype=int)
        self.sd_post = np.zeros((blocks, T, self.ns, self.nd))
        self.sd_post[:, 0] = sd_prior
        self.ab_post = np.zeros((blocks, T, self.ns, 2))
        self.ab_post[:, 0] = ab_prior
        self.pi_post = np.zeros((blocks, T, self.ns))

    def update_beliefs(self, t, observations, responses):
        # update beliefs about hidden states
        blocks = list(range(len(observations)))

        sd_prior = self.sd_post[:, t]
        ab_prior = self.ab_post[:, t]

        # reformat observation vector
        o = np.vstack([observations, 1-observations]).T

        # outcome count and outcome expectations
        nus = ab_prior.sum(axis=-1)
        mus = ab_prior/nus[:, :, None]

        # likelihood of no reversal state
        ollNR = np.sum(np.log(mus[blocks, responses])*o, axis=-1)

        # likelihood of reversal state
        ollR = np.sum(np.log(mus[blocks, 1 - responses])*o, axis=-1)

        # marginal probability \tilde{p}(s_t)
        marg = sd_prior.sum(axis=-1)
        # conditional probability \tilde{p}(d_t|s_t)
        cond = sd_prior/marg[:, :, None]

        theta = marg[:, 0]
        theta = theta/(theta + np.exp(ollR-ollNR)*(1-theta))

        # estimate posterior state distribution
        sd_post = cond*np.vstack([theta, 1-theta]).T[:, :, None]

        ab_post = ab_prior.copy()
        ab_post[blocks, responses] += o*theta[:, None]
        ab_post[blocks, 1-responses] += o*(1-theta[:, None])

        if t+1 < self.sd_post.shape[1]:
            self.sd_post[:, t+1] = np.einsum('ijkl, mkl -> mij',
                                             self.trans_matrix,
                                             sd_post)
            self.ab_post[:, t+1] = ab_post

    def generate_responses(self, t):
        thetas = self.sd_post[:, t].sum(axis=-1)[:, 0]
        mus = self.ab_post[:, t] / self.ab_post[:, t].sum(axis=-1)[:, :, None]

        # compute expected reward probability for each choice
        exp_rew = thetas[:, None] * mus[:, :, 0]
        exp_rew[:, 0] += (1-thetas) * mus[:, 1, 0]
        exp_rew[:, 1] += (1-thetas)*mus[:, 0, 0]

        values = 2*exp_rew - 1

        # get response probability
        dv = values.T-values.max(axis=-1)
        res_prob = np.exp(dv/self.tau)
        res_prob /= res_prob.sum(axis=0)

        return (np.random.rand(self.blocks) > res_prob[0]).astype(int)


class RLReversalLearner(object):

    def __init__(self, alpha, kappa, tau=1e-10,
                 init_values=None,
                 number_of_choices=2,
                 blocks=1, T=100,
                 **kwargs):

        self.blocks = blocks
        self.T = T

        self.alpha = alpha
        self.kappa = kappa
        self.tau = tau

        # set initial value vector
        self.values = np.zeros((blocks, T, number_of_choices))
        if init_values is not None:
            self.values[:, 0] = init_values

    def update_beliefs(self, t, observations, responses):
        # update choice values
        values = self.values[:, t]

        blocks = list(range(self.blocks))

        o = 2*observations - 1
        v1 = values[blocks, responses]
        v2 = values[blocks, 1-responses]
        if t+1 < self.values.shape[1]:
            self.values[:, t+1] = values
            self.values[blocks, t + 1, responses] = v1 + self.alpha*(o - v1)
            self.values[blocks, t + 1, 1 - responses] = v2 + self.alpha*self.kappa*(-o - v2)

    def generate_responses(self, t):
        # get response probability
        dv = self.values[:, t].T - self.values[:, t].max(axis=-1)
        res_prob = np.exp(dv/self.tau)
        res_prob /= res_prob.sum(axis=0)

        return (np.random.rand(self.blocks) > res_prob[0]).astype(int)
