#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot subjects' beliefs over interval durations
@author: Dimitrije Markovic
"""

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.special import gammaln

sns.set(context='talk', style = 'white', color_codes = True)
sns.set_style('ticks')

store = pd.HDFStore('results.h5')
data = store['edhmm/trace']
store.close()


n = data.shape[0]
subjects = [3,4,8,9,11,18]
mu = np.zeros((n,len(subjects)))
sigma = np.zeros((n,len(subjects)))

rs = data.loc[:, 'r__0':'r__21']
ss = data.loc[:, 's__0':'s__21']

nd = 200
k = np.arange(nd)
def neg_binom(k, r, h):
    binomln = gammaln(k+r) - gammaln(k+1) - gammaln(r)
    lp = binomln + k*np.log(1-h) + (r)*np.log(h)
    p = np.exp(lp - lp.max())
    
    return p/p.sum()

pd = np.zeros((n, nd, len(subjects)))
for i in range(n):
    for j in range(len(subjects)):
        h = ss.loc[i,'s__%d' % subjects[j]]
        r = rs.loc[i,'r__%d' % subjects[j]]
        pd[i,:,j] = neg_binom(k, r, h)

mpd = np.zeros((nd, len(subjects)))
for j in range(len(subjects)):
    mpd[:,j] = neg_binom(k, rs.loc[:,'r__%d' % subjects[j]].mean(), 
                    ss.loc[:,'s__%d' % subjects[j]].mean())

fig, ax = plt.subplots(2, 3, figsize = (10, 6), sharex = True, sharey = True)

axs = ax.flatten()

T = 50
d = np.arange(1,T+1)
for j, ax in enumerate(axs):
    for i in range(50):
        ax.plot(np.arange(1,T+1), pd[i, :T, j], 'b', alpha = 0.1)
        ax.plot(np.arange(1,T+1), mpd[:T, j], 'k', linewidth = 1.)
        ax.set_title('Participant %d' % subjects[j])
        ax.set_xlim([1,T])
        if j>2:
            ax.set_xlabel(r'$d$')
        if j == 0 or j == 3:
            ax.set_ylabel(r'$p_0(d)$')

fig.savefig('Fig10.pdf', bbox_inches = 'tight', transparent = True)