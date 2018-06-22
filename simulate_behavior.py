from __future__ import division, print_function
from builtins import range
import numpy as np

from model.simulator import Simulator
from model.tasks import RevLearn
from model.agents import BayesianReversalLearner, RLReversalLearner

from stats import neg_binomial
from plotting import plot_stats

def get_choice_probability(correct, reversals):
    T = reversals.shape[-1]
    suma = np.zeros(21)
    count = 0
    for t in range(T):
        n_rev = reversals[:,t].sum()
        if n_rev > 0 and t > 8 and t < T-11:
            suma += correct[reversals[:,t],t-9:t+12].sum(axis = 0)
            count += n_rev
    
    return suma/count

def generate_behavior(agent_label, task, mu, d, state_transition_matrix):
    
    if agent_label == 'DU':
        alpha = 0.25
        kappa = 1
        
        agent = RLReversalLearner(alpha, kappa, blocks = blocks, T = T)
    
    elif agent_label == 'SU':
        alpha = 0.25
        kappa = 0
        
        agent = RLReversalLearner(alpha, kappa, blocks = blocks, T = T)

    else:
        if agent_label == 'ED-HMM':
            sigma = mu
        else:
            sigma = mu*(mu-1)
        
        #define transition matrix of state durations
        duration_transition_matrix = np.zeros((d_max, d_max))
    
        duration_transition_matrix[:,0] = neg_binomial(mu, sigma, d) 
        
        for k in range(1,len(d)):
            duration_transition_matrix[k-1, k] = 1
        
        #define state-duration transition matrix
        transition_matrix = np.einsum('ijk, lk-> iljk',
                                  state_transition_matrix,
                                  duration_transition_matrix)
    
        prior_states = {}
        psd = np.zeros((ns, d_max)) #prior over state durations
        psd[:,0] = 1
        psd /= psd.sum()
        # set the same state duration priors over all experimental blocks (simulated subjects)
        prior_states['sd'] = np.asarray([psd]*blocks)
        beta_params = 8*np.eye(2) + 2*(np.ones((2,2)) - np.eye(2))
        prior_states['ab'] = np.asarray([beta_params]*blocks)
    
        agent = BayesianReversalLearner(transition_matrix,
                                        prior_states = prior_states, 
                                        blocks = blocks, 
                                        T = T)
    
    sim = Simulator(task, agent, blocks = blocks, T = T).simulate_experiment()
    
    correct_choices = 1-np.logical_xor(task.hidden_states[:,:,0], sim.responses) #performance
    responses = sim.responses
    hidden_states = task.hidden_states    
    
    return correct_choices, responses, hidden_states
    

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
state_transition_matrix[:,:,1:] = np.eye(ns)[:,:,np.newaxis]

#############Simulate ED-HMM and HMM agents####################################

performance = np.zeros((2,2,21,blocks))
choice_prob = np.zeros((2,2,21,21))
for i,env in enumerate(['irregular', 'semi-regular']):
    mu = 20
    if env == 'irregular':
        sigma = mu*(mu-1)
    else:
        sigma = mu
        
    D = neg_binomial(mu, sigma, d)
    
    #set task environments with fixed hidden states within each block
    task = RevLearn(O, S, D, blocks = blocks, T = T)
    task.set_hidden_states()
    
    for j,q in enumerate(['HMM', 'ED-HMM']):
        for k, mean in enumerate(np.arange(10,31)):
            correct_choices, responses, hidden_states = \
                generate_behavior(q, task, mean, d, state_transition_matrix)
            
            performance[i,j,k] = correct_choices.mean(axis = -1)
            choice_prob[i,j,k] = get_choice_probability(correct_choices, 
                                                   hidden_states[:,:,1] == 0)

#plot stats    
fig1 = plot_stats(performance, choice_prob)

###############################################################################


####################run simulations for all agents############################# 
labels = np.array(['HMM', 'ED-HMM', 'SU', 'DU'])
performance = np.zeros((2,len(labels),blocks))
choice_prob = np.zeros((2,len(labels),21))

for i,s in enumerate(['irregular', 'semi-regular']):
    mu = 20
    if s == 'irregular':
        sigma = mu*(mu-1)
    else:
        sigma = mu
    
    D = neg_binomial(mu, sigma, d)

    #set task environments with fixed hidden states within each block
    task = RevLearn(O, S, D, blocks = blocks, T = T)
    task.set_hidden_states()

    for j,label in enumerate(labels):
        correct_choices, responses, hidden_states =\
            generate_behavior(label, task, mu, d, state_transition_matrix)
        
        performance[i,j] = correct_choices.mean(axis = -1)
        choice_prob[i,j] = get_choice_probability(correct_choices, 
                                                  hidden_states[:,:,1] == 0)



##plot stats
fig2 = plot_stats(performance, choice_prob, boxplot=True)

