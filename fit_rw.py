#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit dual update RW model
@author: Dimitrije Markovic
"""

from os import walk

from pymc3.distributions.timeseries import scan

from theano.tensor import exp,log
import theano.tensor as tt

from pymc3 import Model, fit, trace_to_dataframe
from pymc3 import Deterministic, HalfCauchy, Categorical

from theano_models import rw_model

import numpy as np
import seaborn as sns

import pandas as pd

f = []
for (dirpath, dirnames, filenames) in walk('data'):
    f.extend(filenames)
    break
f = np.sort(f)[1:]

n_subs = len(f) #number of subjects

T = 160 #total number of trials
last = -35 # set the time point for the post-reversal phase
inp = np.zeros((T, n_subs, 3), dtype = np.int64)
nans = np.zeros((T, n_subs), dtype = bool)
#format data
for i,file in enumerate(f):
    sub, _ = file.split(sep='.')
    sub = int(sub[-2:])
    data = pd.read_csv(dirpath + '/' + file)
    start = data['S'][0]
    data = data.loc[:,'A':'R']

    #change choice labels from (1,2) to (0,1)  
    if start == 1:
        data['A'] -= 1
    else:
        #reverse labels (1,2) to (1,0) if the initial best choice was 2
        data['A'] = 2 - data['A']

    data['T'] = 1 #set the switch variable to one (it is set to zero only for nan trials)
    nans[:,i] = data.isnull().any(axis = 1)
    data[nans[:,i]] = 0
    data = data.astype(int)
    inp[:,i,:] = data.values


#define the hierarchical models
with Model() as dual_model:

    #set hierarchical priors for learning rates
    atau = HalfCauchy('atau', beta = 1)
    aloc = HalfCauchy('aloc', beta = atau, shape = (n_subs,))
    alpha = Deterministic('alpha', aloc/(1+aloc)) 
    
    #set hierarchical priors for coupling strengths
    ktau = HalfCauchy('ktau', beta = 1)
    kloc = HalfCauchy('kloc', beta = ktau, shape = (n_subs,))
    kappa = Deterministic('kappa', kloc/(1+kloc))

    #set hierarchical priors for response noises   
    btau = HalfCauchy('btau', beta = 1)
    bloc = HalfCauchy('bloc', beta = btau, shape=(n_subs,))
    beta = Deterministic('beta', 1/bloc)
    
    #set hierarchical priors for initial choice value
    mtau = HalfCauchy('mtau', beta = 1)
    mlocA = HalfCauchy('mlocA', beta = mtau, shape = (n_subs,))
    mlocB = HalfCauchy('mlocB', beta = mtau, shape = (n_subs,))
    muA = Deterministic('muA', mlocA/(1+mlocA)) 
    muB = Deterministic('muB', 1/(1+mlocB)) 
    V0 = tt.stacklists([2*muA -1, 2*muB -1]).T
    
    #compute the choice values
    (Q, _) = scan(rw_model, sequences = [inp],
                 outputs_info = V0,
                 non_sequences = [alpha, kappa, range(n_subs)],
                 name = 'rw')

    V0 = Deterministic('V0', V0[None,:,:])
    V = Deterministic('V', tt.concatenate([V0, Q[:-1]]))
    
    #set hierarchical prior for response biases
    ctau = HalfCauchy('ctau', beta = 1)
    cloc = HalfCauchy('cloc', beta = ctau, shape = (n_subs,))
    c0 = Deterministic('c0', cloc/(1+cloc))

    #compute response noise and response bias modulated response values
    G = Deterministic('G', beta[None,:, None]*V + log([c0, 1-c0]).T[None, :, :])
    
    #compute response probability for the prereversal and the reversal phase of the experiment
    nzero = tt.nonzero(~nans[:last])
    p = Deterministic('p', tt.nnet.softmax(G[:last][nzero]))

    #set observation likelihood of responses
    observed = Categorical('obs', p = p, observed = inp[:last,:,0][~nans[:last]])

#fit the model    
with dual_model:
     approx = fit(method = 'advi', n = 50000, progressbar = True)

#sample from posterior
nsamples = 1000
trace = approx.sample(nsamples, include_transformed = True)

data = trace_to_dataframe(trace, include_transformed=True, 
                          varnames=['ctau', 'c0', 
                                    'mtau', 'muA', 'muB', 
                                    'btau', 'beta',
                                    'ktau', 'kappa', 
                                    'atau', 'alpha'])

#get posterior samples of the predicted response value in the postreversal phase
Gs = np.zeros((10000, 35, n_subs, 2))
for i in range(10):
    trace = approx.sample(nsamples, include_transformed = True)
    Gs[i*1000:(i+1)*1000] = trace['G'][:,last:]

post_like = np.exp(Gs-Gs.max(axis=-1)[:,:,:,None])
post_like /= post_like.sum(axis = -1)[:,:,:,None]

#get response outcomes in the post-reversal phase
obs = inp[None, last:, :, 0]

#compute observation likelihood for each posterior sample
post_ol = post_like[:,:,:,0]**(1-obs)*post_like[:,:,:,1]**obs

#compute mean likelihood over all samples
mean_pl = post_ol.prod(axis = 1).mean(axis = 0)

#get posterior predictive log model evidence
pplme = np.log(mean_pl)

#get mean probability of selecting stimuli 2
plike2 = post_like[:,:,:,1].mean(axis = 0)

store = pd.HDFStore('results.h5')
store['durw/trace'] = data
store['durw/pplme'] = pd.Series(pplme)
store['durw/plike2'] = pd.DataFrame(plike2)
store.close()