"""This module contains the Simulator class that defines interactions between
the environment and the agent. It also keeps track of all generated observations
and responses generated. To initiate it one needs to provide the environment
class and the agent class that will be used for the experiment.
"""
from numpy import zeros


class Simulator(object):

    def __init__(self, environment, agent, blocks=1, T=10):

        # set inital elements of the world to None
        self.environment = environment
        self.agent = agent

        self.blocks = blocks  # number of experimental blocks
        self.T = T  # number of trials in each block

        # container for observations
        self.observations = zeros((self.blocks, self.T))-1

        # container for agents responses
        self.responses = zeros((self.blocks, self.T), dtype=int) - 1

    def simulate_experiment(self):
        """Runs the experiment by iterating through all the blocks and trials.
           Here we generate responses and outcomes and update the agent's beliefs.
        """
        for t in range(self.T):
            # update single trial
            res = self.agent.generate_responses(t)

            obs = self.environment.generate_observations(t, res)

            self.agent.update_beliefs(t, obs, res)

            self.observations[:, t] = obs
            self.responses[:, t] = res

        return self
