#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for plotting posterior estimate of prior beliefs over between
reversal intervals, effective change probability, and the probability that 
the posterior estimate of r is larger than 1.5.
"""
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from stats import neg_binomial, evolve_change_prob, kl_div

sns.set(context='talk', style = 'white', color_codes = True)
sns.set_style('ticks')

store = pd.HDFStore('behavior_fits.h5')
data = store['edhmm/trace']
store.close()


#define which subjects out of 22 to plot 
n_subs = 22
n_samples = data.shape[0]
subjects = np.array([3,4,8,9,11,18])


#make duration vector
d_max = 200
d = np.arange(d_max)


rs = data.loc[:, 'r__0':'r__21'] #posterior samples of r
deltas = data.loc[:, 'dlt__0':'dlt__21'] #posterior samples of delta

p0d = np.zeros((n_samples, n_subs, d_max))
divs = np.zeros((n_samples, n_subs))
for i in range(n_samples):
    for j in range(n_subs):
        dlt = deltas.loc[i,'dlt__%d' % j]
        r = rs.loc[i,'r__%d' % j]
        p0d[i,j] = neg_binomial(dlt,r,d,transformed = False)
        
        geometric = neg_binomial(dlt,1,d,transformed = False)
        divs[i,j] = kl_div(geometric, p0d[i,j])

fig1, ax = plt.subplots(1,1, figsize = (12,3), sharex=True)   
probs = np.sum(rs.values > 1.5, axis=0)/n_samples
ax.bar(np.arange(1,23), probs, color = '#72B2F2')

ax.set_xticks(np.arange(1,23))
       
ax.set_xlim([.5, 22.5])
ax.set_ylabel(r'$Pr(r>1.5)$')
ax.set_xlabel('participant')
sns.despine(fig = fig1)



fig1.savefig('Fig13.pdf', bbox_inches = 'tight', transparent = True)

mpd = np.zeros((d_max, n_subs))
for j in range(n_subs):
    mpd[:,j] = neg_binomial(np.mean(deltas.loc[:,'dlt__%d' % j]),
                            np.mean(rs.loc[:,'r__%d' % j]),
                            d,transformed = False)

fig2, ax = plt.subplots(2, 3, figsize = (10, 6), sharex = True, sharey = True)

axs = ax.flatten()

T = 50
durs = np.arange(1,T+1)
for j, ax in enumerate(axs):
    for i in range(50):
        ax.plot(durs, p0d[i,subjects[j]-1,:T], 'b', alpha = 0.1)
        ax.plot(durs, mpd[:T, subjects[j]-1], 'k', linewidth = 1.)
        ax.set_title('Participant %d' % subjects[j])
        ax.set_xlim([1,T])
        if j>2:
            ax.set_xlabel(r'$d$')
        if j == 0 or j == 3:
            ax.set_ylabel(r'$p_0(d)$')

#fig2.savefig('Sup_Fig1.pdf', bbox_inches = 'tight', transparent = True)
            
fig2, ax = plt.subplots(2, 3, figsize = (10, 6), sharex = True, sharey = True)
axs = ax.flatten()

ds = 2
R = (np.ones((ds,ds)) - np.eye(ds))/(ds-1)
S = np.zeros((ds,ds,d_max))
S[:,:,0] = R
S[:,:,1:] = np.eye(ds)[:,:,np.newaxis]

D = np.zeros((d_max, d_max))
for k in range(1,d_max):
    D[k-1, k] = 1

T = 40
durs = np.arange(1,T+1)
for j, ax in enumerate(axs):
    for i in range(50):
        D[:,0] = p0d[i,subjects[j]-1]
        ax.plot(durs, evolve_change_prob(S, D, T+1), 
             'b', alpha = .1)
    D[:,0] = mpd[:, subjects[j]-1]
    ax.plot(durs, evolve_change_prob(S, D, T+1), 'k', linewidth = 2.)
    
    ax.set_title('Participant %d' % subjects[j])
    ax.set_xlim([1,T])
    if j>2:
        ax.set_xlabel(r'$\tau$')
    if j == 0 or j == 3:
        ax.set_ylabel(r'$\delta_\tau$')
        
fig2.savefig('Fig12.pdf', bbox_inches = 'tight', transparent = True)