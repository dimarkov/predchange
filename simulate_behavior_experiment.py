from __future__ import division, print_function
from builtins import range
import numpy as np
import pandas as pd

from model.simulator import Simulator
from model.tasks import RevLearn
from model.agents import BayesianReversalLearner, RLReversalLearner

from stats import neg_binomial

def generate_behavior(agent_label, task, state_transition_matrix):
    
    if agent_label == 'DU-RW':
        alpha = .25
        kappa = 1.
        
        agent = RLReversalLearner(alpha, kappa, tau=.25, blocks = blocks, T = T)
    
    elif agent_label == 'SU-RW':
        alpha = .25
        kappa = 0
        
        agent = RLReversalLearner(alpha, kappa, tau=.25, blocks = blocks, T = T)

    else:
        if agent_label == 'RRI':
            delta = 1/6
            r = 20*delta/(1-delta)
        else:
            r = 1
            delta = 0.05
        
        #define transition matrix of state durations
        duration_transition_matrix = np.zeros((d_max, d_max))
    
        duration_transition_matrix[:,0] = neg_binomial(delta, r, d, transformed = False) 
        
        for k in range(1,len(d)):
            duration_transition_matrix[k-1, k] = 1
        
        #define state-duration transition matrix
        transition_matrix = np.einsum('ijk, lk-> iljk',
                                  state_transition_matrix,
                                  duration_transition_matrix)
    
        prior_states = {}
        psd = np.zeros((ns, d_max)) #prior over state durations
        psd[:,:] = duration_transition_matrix[:,0][None,:]
        psd /= psd.sum()
        # set the same state duration priors over all experimental blocks (simulated subjects)
        prior_states['sd'] = np.asarray([psd]*blocks)
        beta_params = 8*np.eye(2) + 2*(np.ones((2,2)) - np.eye(2))
        prior_states['ab'] = np.asarray([beta_params]*blocks)
    
        agent = BayesianReversalLearner(transition_matrix,
                                        prior_states = prior_states,
                                        tau = 0.25,
                                        blocks = blocks, 
                                        T = T)
    
    sim = Simulator(task, agent, blocks = blocks, T = T).simulate_experiment()
    
    
    responses = sim.responses
    outcomes = sim.observations
    correct_choices = 1-np.logical_xor(task.hidden_states[:,:,0], responses) #performance
    
    return correct_choices, responses, outcomes
    

blocks = 1000 #number of experimental blocks for each model
T = 160 #number of trials for each experimental block

ns = 2 #number of states
no = 2 #number of observations
na = 2 #number of choices
d_max = 100 #maximal interval duration
d = np.arange(d_max) #vector of possible durations

world_params = {'T': T, 'blocks': blocks, 'ns': ns,
                'no': no, 'na': na, 'd_max': d_max}

#define reward probabilities
pH = 0.8
pL = 0.2

#define observation likelihood
O = pH*np.eye(ns) + pL*(np.ones((ns,ns)) - np.eye(ns))
observation_likelihood = np.zeros((no, ns, na))
observation_likelihood[0, :,:] = 1-O
observation_likelihood[1, :,:] = O

#define state transition matrix
S = (np.ones((ns,ns)) - np.eye(ns))/(ns-1)
state_transition_matrix = np.zeros((ns,ns,d_max))
state_transition_matrix[:,:,0] = S
state_transition_matrix[:,:,1:] = np.eye(ns)[:,:,None]

# duration probability is not relevant for simulations, fixed to unifrom dist.
D = np.ones(T)/T

store = pd.HDFStore('behavior_fits.h5')
hidden_states = store['hidden_states'].values
store.close()

#set task environments with fixed hidden states within each block
task = RevLearn(O, S, D, blocks = blocks, T = T)
task.set_hidden_states(hidden_states=hidden_states[None,:, None])

####################run simulations for all agents############################# 
labels = np.array(['IRI', 'RRI', 'SU-RW', 'DU-RW'])
performance = np.zeros((2,len(labels),blocks))
choice_prob = np.zeros((2,len(labels),21))

mean_response = np.zeros((T, 4))
subsample = 25
behavior = np.zeros((T, 4*subsample, 2))

for i, label in enumerate(labels):
    
    correct_choices, responses, outcomes =\
            generate_behavior(label, task, state_transition_matrix)
    
    locs = np.random.randint(0, high = blocks, size = subsample)
    behavior[:, i*subsample:(i+1)*subsample, 0] = responses.T[:,locs]
    behavior[:, i*subsample:(i+1)*subsample, 1] = outcomes.T[:,locs]
    
    mean_response[:,i] =  correct_choices.mean(axis = 0)   
    print(label,' ', np.median(correct_choices.mean(axis = -1)))

#uncomment to overwite the files with new data
#np.save('mean_responses_exp.npy', mean_response)
#np.save('simulated_behavior.npy', behavior)