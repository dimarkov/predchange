#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from pymc3 import Model, fit, logsumexp, traceplot, sample, plot_posterior
from pymc3 import DensityDist, Dirichlet, Bernoulli, Deterministic, HalfCauchy, Gamma, Categorical, InverseGamma, Beta

import theano.tensor as tt

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(context='talk', style = 'white', palette='muted', color_codes = True)

#load the modle fitting results
store = pd.HDFStore('results.h5')

num_sub = 22 #number of subjects
LME = pd.DataFrame(index = range(num_sub))

cols = ['DU-RW', 'ED-HMM']

LME['DU-RW'] = store['eat/rl/ppc']
LME['ED-HMM'] = store['eat/edhmm/ppc']
store.close()

N, M = LME.shape
nsample = 100000

# define model mixture log-likelihood 
def logp_mix(mf):
    def logp_(value):
        logps = tt.log(mf) + value

        return tt.sum(logsumexp(logps, axis=1))

    return logp_


# define and fit the probabilistic model
with Model() as model:
    tau = HalfCauchy('tau', beta = 1)
    mf = Dirichlet('mf', a = tt.ones(M)/tau, shape=(M,))
    xs = DensityDist('logml', logp_mix(mf), observed=LME)
    approx = fit(method='advi', n = 20000)
    
trace = approx.sample(nsample)    
traceplot(trace);

#compute exceedance probability
ep, _ = np.histogram(trace['mf'].argmax(axis = 1), bins = M)
ep = pd.DataFrame({'ep':ep/nsample, 'models': cols})


fig = plt.figure(figsize = (10,5))
ax1 = plt.subplot(121)
ax3 = plt.subplot(122)

# plot pdf of posterior model probability
sns.kdeplot(trace['mf'][:,1], color = 'b', shade = True, ax = ax1);
ax1.vlines(.5, ymin = 0, ymax = 2.75, color = 'k', linestyle = '--', alpha = 0.5, zorder = 1);
ax1.set_ylim([0,3])
sns.despine(ax = ax1)
ax1.set_ylabel('density')
ax1.set_xlabel('posterior probability')
ax1.text(-.35,3,'A')

# compute model attribution
nlme = LME + np.log(trace['mf']).mean(axis = 0)
p = np.exp(nlme.values - nlme.max(axis = 1)[:, np.newaxis]);
p /= p.sum(axis = 1)[:,np.newaxis]

# plot model attribution
index = pd.Index(data = np.arange(1,len(p)+1), name = 'participant')
columns = pd.Index(data = cols, name = 'models')
dp = pd.DataFrame(data = p, index = index, columns = columns)
sns.heatmap(dp, vmin=0, vmax=1, cmap = 'viridis', ax = ax3);
ax3.set_title('model attribution');
ax3.text(-.35,22, 'B')
fig.savefig('mc.pdf', bbox_inches = 'tight', transparent = True)