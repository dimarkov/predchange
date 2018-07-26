#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model comparison of dual update RW and ED-HMM models
@author: Dimitrije Markovic
"""
import pandas as pd
import numpy as np

from pymc3 import Model, fit, logsumexp, traceplot
from pymc3 import DensityDist, Dirichlet, HalfCauchy

import theano.tensor as tt

import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

sns.set(context='talk', style = 'white', palette='muted', color_codes = True)

simulated = True #model comparison for simulated data

#load the modle fitting results
if simulated:
    store = pd.HDFStore('simulation_fits.h5')
else:
    store = pd.HDFStore('behavior_fits.h5')

#uncomment when comparing models on simulated data

cols = ['DU-RW', 'ED-HMM']
LME = pd.DataFrame(columns = cols)


LME['DU-RW'] = store['durw/pplme']
LME['ED-HMM'] = store['edhmm/pplme']
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
    tau = HalfCauchy('tau', beta = 1.)
    
    mf = Dirichlet('mf', a = tt.ones(M)/tau, shape=(M,))
    xs = DensityDist('logml', logp_mix(mf), observed=LME)

with model:    
    approx = fit(method='advi', n = 10000)
    
trace = approx.sample(nsample)    
traceplot(trace);

#compute exceedance probability
ep, _ = np.histogram(trace['mf'].argmax(axis = 1), bins = M)
ep = pd.DataFrame({'ep':ep/nsample, 'models': cols})


fig = plt.figure(figsize = (10,5))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

# plot pdf of posterior model probability
sns.kdeplot(trace['mf'][:,1], color = 'b', shade = True, ax = ax1);
ax1.vlines(.5, ymin = 0, ymax = 2.75, color = 'k', linestyle = '--', alpha = 0.5, zorder = 1);
sns.despine(ax = ax1)
ax1.set_ylabel('density')
ax1.set_xlabel('posterior probability')

# compute model attribution
nlme = LME + np.log(trace['mf']).mean(axis = 0)
p = np.exp(nlme.values - nlme.max(axis = 1)[:, np.newaxis]);
p /= p.sum(axis = 1)[:,np.newaxis]

# plot model attribution
index = pd.Index(data = np.arange(1,len(p)+1), name = 'participant')
columns = pd.Index(data = cols, name = 'models')
dp = pd.DataFrame(data = p, index = index, columns = columns)
sns.heatmap(dp, vmin=0, vmax=1, cmap = 'viridis', ax = ax2);
ax2.set_title('model attribution');
ax2.text(-.35,N, 'B')

# mark subjects with higher attribution from ED-HMM
if not simulated:
    ax1.set_ylim([0,3])
    ax1.text(-.35,3,'A')

    rects = []
    for y in [4,11]:
        rects.append(Rectangle((0, y),2,1,fill=False,alpha=1,color='r',lw =3))
    for y in [13,18]:
        rects.append(Rectangle((0, y),2,2,fill=False,alpha=1,color='r',lw =3))
    pc = PatchCollection(rects, facecolor='none',edgecolor='r',lw=2)
    ax2.add_collection(pc)

    fig.savefig('Fig9.pdf', bbox_inches = 'tight', transparent = True)
else:
    ax1.set_ylim([0,3.5])
    ax1.text(-.35,3.5,'A')

    ax2.set_yticks([1, 10, 20, 30, 40, 50, 60, 70, 80], minor = False)
    ax2.set_yticklabels(['80',  '70', '60', '50', '40', '30', '20', '10', '1'])
    
    print('number of simulated subjects classified as ED-HMM:', \
          (p[:,0] < p[:,1]).reshape(4,-1).sum(axis = -1))