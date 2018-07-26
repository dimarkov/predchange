import numpy as np

from stats import neg_binomial

from model.tasks import RevLearn

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set(context = 'talk', style = 'ticks', color_codes=True)

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

from plotting import plot_between_reversal_interval_distribution

T = 160 #number of trials

ds = 2 #number of states
do = 2 #number of observations
da = ds #number of choices
d_max = 100 #maximal interval duration 

world_params = {'T': T, 'ds': ds,
                'do': do, 'da': da, 'd_max': d_max}

def generate_series(env):
    #make observation matrix
    pH = 0.8
    pL = 0.2
    O = pH*np.eye(ds) + pL*(np.ones((ds,ds)) - np.eye(ds))
    
    #make state transition matrix
    S = (np.ones((ds,ds)) - np.eye(ds))/(ds-1)
    
    #make duration distribution
    mu = 20
    if env == 'geo':
        sigma = mu*(mu-1)
    else:
        sigma = mu
    
    d = np.arange(d_max)
    D = neg_binomial(mu, sigma, d)
    
    rev_env = RevLearn(O, S, D, T = T)
    rev_env.set_hidden_states()
        
    states = rev_env.hidden_states[0,:,0].astype(bool)
    ol = np.zeros((T,2))
    ol[states,0] = pH
    ol[~states,0] = pL
    ol[states,1] = pL
    ol[~states,1] = pH
    
    return ol

actions = {}    
for env in ['geo', 'uni']:
    actions[env] = generate_series(env)

gs = gridspec.GridSpec(nrows=2, ncols=3)

fig = plt.figure(figsize = (12,6))
fig.subplots_adjust(wspace = .5)
ax0 = fig.add_subplot(gs[0, :2])
ax1 = fig.add_subplot(gs[1, :2])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 2])
ax = [ax0, ax1, ax2, ax3]


plot_between_reversal_interval_distribution(ax[2:])
ax[2].set_xticklabels([])

ax[2].text(50, .08, 'irregular reversals', rotation = -90, fontsize = 14)
ax[3].text(50, .1, 'semi-regular reversals', rotation = -90, fontsize = 14)

sns.set_palette(flatui)
ax[0].plot(np.arange(1,T+1), actions['geo'])
ax[1].plot(np.arange(1,T+1), actions['uni'])

ax[0].set_ylabel('reward probability', fontsize = 12)
ax[1].set_ylabel('reward probability', fontsize = 12)
ax[1].set_xlabel('trial')

for i in range(2):
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].set_ylim([0, 1])
    ax[i].set_xlim([1, T])
ax[0].set_xticklabels([])
    
ax[0].text(120, .7, 'Choice A', color = flatui[0])
ax[0].text(120, .22, 'Choice B', color = flatui[1])

#uncomment to save a new figure
#fig.savefig('Fig4.pdf', bbox_inches = 'tight', transparent = True)