from __future__ import division, print_function
from builtins import range
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context = 'talk', style = 'white', color_codes=True)
sns.set_style('ticks')

from stats import neg_binomial, evolve_change_prob

d_max = 200 #maximal interval duration
d = np.arange(d_max)

ds = 2
R = (np.ones((ds,ds)) - np.eye(ds))/(ds-1)
S = np.zeros((ds,ds,d_max))
S[:,:,0] = R
S[:,:,1:] = np.eye(ds)[:,:,np.newaxis]

D = np.zeros((d_max, d_max))
for k in range(1,d_max):
    D[k-1, k] = 1 

fig1, ax1 = plt.subplots(figsize = (10,5))
fig2, ax2 = plt.subplots(figsize = (10,5))

ax1.vlines(20, 0 ,.15, color = 'k', linestyle='--', label = r'$\mu$')
ax2.vlines(20, 0 ,.09, color = 'k', linestyle='--', label = r'$\mu$')

delta_df = pd.DataFrame()
mu = 20
labels = [r'$\sigma = \mu$', r'$\sigma = 10\mu$', r'$\sigma = (\mu-1)\mu$']
for i, x in enumerate([1, 10, 19]):
    sigma = mu*x
    p0 = neg_binomial(mu, sigma, d)
    D[:,0] = p0
    ax2.bar(np.arange(d_max)+1, p0, alpha = 0.5, label = labels[i])
    T = 100
    ax1.plot(np.arange(2, T+1, 1), evolve_change_prob(S, D, T), 
             'd', alpha = 0.8, label = labels[i])


ax1.legend()
ax1.set_xlim([1,80])
ax2.legend()
ax2.set_xlim([1,40])
ax2.set_xlabel('$d$')
ax2.set_ylabel('$p_0(d)$')

ax1.set_ylabel(r'$\delta_{\tau}$')
ax1.set_xlabel(r'$\tau$')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False) 
ax2.spines['right'].set_visible(False)

fig1.savefig('Fig2.pdf', bbox_inches = 'tight', transparent = True)
fig2.savefig('Fig1.pdf', bbox_inches = 'tight', transparent = True)