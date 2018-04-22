from __future__ import division, print_function
from builtins import range
import numpy as np
from scipy.special import binom, gammaln

import pandas as pd

from model.simulator import Simulator
from model.tasks import RevLearn
from model.agents import BayesianReversalLearner, RLReversalLearner

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(context = 'talk', style = 'white', color_codes=True)

def binomln(x1, x2):
    return gammaln(x1+1) - gammaln(x2+1) - gammaln(x1-x2+1)

def fun(x,x0):
    return (x*(d_max*x**(d_max+1) - (d_max+1)*\
               x**d_max + 1)/((1-x)*(1-x**(d_max+1))) - x0)**2

def plot_beliefs(T, states, beliefs_states, beliefs_durations):
    f, (ax1, ax2) = sns.plt.subplots(2, 1, sharex=True)
    
    run_ends = np.arange(1,T+1)[states[:,1] == 0]    
    ax1.plot(np.arange(1, T+1), beliefs_states)
    
    for x in run_ends:
        ax1.axvline(x = x, color = 'r', lw = 2)
        ax2.axvline(x = x, color = 'r', lw = 2)
    
    nx, ny = beliefs_durations.shape
    y, x = np.mgrid[slice(0, ny+1, 1), slice(0, nx+1, 1)]
    ax2.pcolor(x, y, beliefs_durations.T, vmin=0, vmax=1)
    
def plot_controls(T, beliefs_controls, beliefs_policies):
    f, (ax1, ax2) = sns.plt.subplots(2, 1, sharex=True)
    
    ax1.plot(np.arange(1, T+1), beliefs_controls)
    
    nx, ny = beliefs_policies.shape
    y, x = np.mgrid[slice(0, ny+1, 1), slice(0, nx+1, 1)]
    ax2.pcolor(x, y, beliefs_policies.T, vmin=0, vmax=1)
    
    
def generate_behavior(agent_label, task, mu, 
                     d, outcome_utility,
                     state_transition_matrix):
    
    if agent_label == 'ED-HMM':
        sigma = mu
    elif agent_label == 'HMM':
        sigma = mu*(mu-1)
    
    if agent_label == 'DU':
        alpha = 0.25
        kappa = 1
        
        agent = RLReversalLearner(alpha, kappa, blocks = blocks, T = T)
    
    elif agent_label == 'SU':
        alpha = 0.25
        kappa = 0
        
        agent = RLReversalLearner(alpha, kappa, blocks = blocks, T = T)

    else:
        delta = 1-(mu-1)/sigma
        r = (mu-1)**2/(sigma-mu+1)

        #define transition matrix of state durations
        l = binomln(d+r-1, d) + d*np.log(delta) +r*np.log(1-delta)
        dur = np.exp(l - l.max())
        dur = dur/dur.sum()
    
        duration_transition_matrix = np.zeros((d_max, d_max))
    
        duration_transition_matrix[:,0] = dur
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
        prior_states['sd'] = np.asarray([psd]*blocks)
        beta_params = 8*np.eye(2) + 2*(np.ones((2,2)) - np.eye(2))
        prior_states['ab'] = np.asarray([beta_params]*blocks)
    
        agent = BayesianReversalLearner(transition_matrix,
                                        prior_states = prior_states, 
                                        blocks = blocks, 
                                        T = T)
    
    sim = Simulator(task, agent, blocks = blocks, T = T).simulate_experiment()
    
    perf = 1-np.logical_xor(task.hidden_states[:,:,0], sim.responses) #performance
    stay = (sim.responses[:,:-1] == sim.responses[:,1:]) #stay probability
    lose_stay = stay*(sim.observations[:,:-1] == 0) #lose stay probability
        
    return perf.sum(axis = 1)/T, stay.sum(axis = 1)/(T-1), \
           lose_stay.sum(axis=1)/(sim.observations[:,:-1] == 0).sum(axis = 1)
    
blocks = 100 #number of experimental blocks for each model
T = 160 #number of trials for each experimental block

ns = 2 #number of states
no = 2 #number of observations
na = 2 #number of choices
d_max = 200 #maximal interval duration
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

#define outcome utility
outcome_utility = np.array([-1, 1])

label = np.array(['ED-HMM', 'HMM', 'DU', 'SU'])
performance = pd.DataFrame()

#run simulations for all agents
for s in ['irregular', 'semi-regular']:
    mu = 20
    if s == 'irregular':
        sigma = mu*(mu-1)
    else:
        sigma = mu
        
    delta = 1-(mu-1)/sigma
    r = (mu-1)**2/(sigma-mu+1)
    D = binom(d+r-1, d)*delta**d*(1-delta)**r

    D = D/D.sum()
    
    task = RevLearn(O, S, D, blocks = blocks, T = T)
    task.set_hidden_states()

    for j, q in enumerate(label):
        if q == 'DU':
            perf, pstay, lstay = generate_behavior(q, task, [], [], [], [])

            data = {'performance': perf,
                    'stay': pstay,
                    'lose-stay':lstay,
                    'agent': q,
                    'reversals': s,
                    '$\mu$':20 }
            
            performance = performance.append(pd.DataFrame(data))
        if q == 'SU':
            
            perf, pstay, lstay = generate_behavior(q, task, [], [], [], [])
            
            data = {'performance': perf,
                    'stay': pstay,
                    'lose-stay': lstay,
                    'agent': q,
                    'reversals': s,
                    '$\mu$':20 }
            
            performance = performance.append(pd.DataFrame(data))
        else:
            for i, mean in enumerate([10,20,30]):
                perf, pstay, lstay = generate_behavior(q, task, mean, 
                                                d, outcome_utility,
                                                state_transition_matrix)
                data = {'performance': perf,
                        'stay': pstay,
                        'lose-stay': lstay,
                        'agent': q,
                        'reversals': s,
                        '$\mu$':mean }
    
                performance = performance.append(pd.DataFrame(data)) 

#plot results
sns.set_style('ticks')

data_loc = np.any([performance['agent'] == 'ED-HMM', \
                   performance['agent'] == 'HMM'], axis = 0)    
g = sns.FacetGrid(performance[data_loc], 
                  col = 'reversals', hue = 'agent', size = 5, 
                  sharex = True, sharey = True);
                  
sns.boxplot(x = '$\mu$', y = 'performance', 
            data = g.data[g.data['reversals']=='irregular'], 
            hue = 'agent', ax = g.axes[0,0])


sns.boxplot(x = '$\mu$', y = 'performance', 
            data = g.data[g.data['reversals']=='semi-regular'], 
            hue = 'agent', ax = g.axes[0,1])

g.axes[0,0].legend().set_visible(False)
g.axes[0,0].set_title('irregular reversals')
g.axes[0,1].get_yaxis().set_visible(False)
g.axes[0,1].set_title('semi-regular reversals')
g.set_xlabels('$\mu$')
sns.despine(offset = 10, trim = True)
g.savefig('Fig5.pdf', bbox_inches = 'tight')

fig = plt.figure(figsize = [10,5])
data = performance[performance['$\mu$'] == 20]
sns.boxplot(x = 'agent', y = 'performance', data = data, hue = 'reversals')
sns.despine(offset = 10, trim = True)
fig.savefig('Fig6.pdf', bbox_inches = 'tight')
