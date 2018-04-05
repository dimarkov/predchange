#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit explicit duration hidden markov model
@author: dimitrije
"""

from os import walk

from pymc3.distributions.timeseries import scan
from pymc3.distributions.special import psi, gammaln

from theano.tensor import exp,log
import theano.tensor as tt

from pymc3 import Model, fit, trace_to_dataframe
from pymc3 import Categorical, HalfCauchy, Deterministic

from theano_models import edhmm_model

import numpy as np
import pandas as pd

f = []
for (dirpath, dirnames, filenames) in walk('data'):
    f.extend(filenames)
    break
f = np.sort(f)[1:]

n_subs = len(f) #number of subjects

d_max = 200 #maximal value for state duration

T = 160 #total number of trials
last = -35 # set the time point for the post-reversal phase
inp = np.zeros((T, n_subs, 3), dtype = np.int64)
nans = np.zeros((T, n_subs), dtype = bool)
#format data
for i,file in enumerate(f[:22]):
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
      
    data['R'][data['R']==-1] = 0 #map choice outcomes from {-1,1} to {0,1}
    data['T'] = 1 #set the switch variable to one (it is set to zero only for nan trials)
    nans[:,i] = data.isnull().any(axis = 1)
    data[nans[:,i]] = 0
    data = data.astype(int)
    inp[:,i,:] = data.values


#define the hierarchical models
with Model() as edhmm:
    d = tt.arange(d_max) #vector of possible duration values from zero to d_max
    d = tt.tile(d, (n_subs,1))
    P = tt.ones((2,2)) - tt.eye(2) #permutation matrix
    
    #set prior state probability
    theta0 = tt.ones(n_subs)/2  
    
    #set hierarchical prior for delta parameter of prior beliefs p_0(d)
    dtau = HalfCauchy('dtau', beta = 1)
    dloc = HalfCauchy('dloc', beta = dtau, shape = (n_subs,))
    delta = Deterministic('delta', dloc/(1+dloc)) 
    
    #set hierarchical prior for r parameter of prior beleifs p_0(d)
    rtau = HalfCauchy('rtau', beta = 1)
    rloc = HalfCauchy('rloc', beta = rtau, shape = (n_subs,))
    r = Deterministic('r', 1+rloc)
    
    #compute prior beliefs over state durations for given
    binomln = tt.gammaln(d+r[:,None]) - tt.gammaln(d+1) - tt.gammaln(r[:,None])
    pd0 = tt.nnet.softmax(binomln + d*log(1-delta[:,None]) + r[:,None]*log(delta[:,None]))
    
    #set joint probability distribution
    joint0 = tt.stack([theta0[:,None]*pd0, (1-theta0)[:,None]*pd0]).dimshuffle(1,0,2)
    
    #set hierarchical priors for response noises   
    btau = HalfCauchy('btau', beta = 1)
    bloc = HalfCauchy('bloc', beta = btau, shape=(n_subs,))
    beta = Deterministic('beta', 1/bloc)
    
    #set hierarchical priors for initial inital beliefs about reward probability
    mtau = HalfCauchy('mtau', beta = 4)
    mloc = HalfCauchy('mloc', beta = mtau, shape = (n_subs,2))
    muA = Deterministic('muA', mloc[:,0]/(1+mloc[:,0])) 
    muB = Deterministic('muB', 1/(1+mloc[:,1])) 
    init = tt.stacklists([[10*muA, 10*(1-muA)], \
                          [10*muB, 10*(1-muB)]]).dimshuffle(2,0,1)
    
    #compute the posterior beleifs over states, durations, and reward probabilities
    (post, _) = scan(edhmm_model, 
                        sequences = [inp],
                        outputs_info = [init, joint0],
                        non_sequences = [pd0, P, range(n_subs)],
                        name = 'edhmm')
    
    #get posterior reward probabliity and state probability
    a0 = init[None,:,:,0]
    b0 = init[None,:,:,1]
    a = tt.concatenate([a0, post[0][:-1,:,:,0]])
    b = tt.concatenate([b0, post[0][:-1,:,:,1]])
    mu = Deterministic('mu', a/(a+b))
    theta = Deterministic('theta', tt.concatenate([theta0[None, :], \
                          post[1][:-1].sum(axis=-1)[:,:,0]])[:,:,None])
    
    #compute choice dependend expected reward probability
    mean = (theta*mu + (1-theta)*mu.dot(P))
    
    #compute expected utility
    U = Deterministic('U', 2*mean-1)

    #set hierarchical prior for response biases
    ctau = HalfCauchy('ctau', beta = 1)
    cloc = HalfCauchy('cloc', beta = ctau, shape = (n_subs,))
    c0 = Deterministic('c0', cloc/(1+cloc))

    #compute response noise and response bias modulated expected free energy
    G = Deterministic('G', beta[None,:, None]*U + log([c0, 1-c0]).T[None, :, :])
    
    #compute response probability for the prereversal and the reversal phase of the experiment
    nzero = tt.nonzero(~nans[:last])
    p = Deterministic('p', tt.nnet.softmax(G[:last][nzero]))
    
    #set observation likelihood of responses
    responses = inp[:last,:,0][~nans[:last]]
    observed = Categorical('obs', p = p, observed = responses)

#fit the model    
with edhmm:
    approx = fit(method = 'advi', n = 5000, progressbar = True)

#sample from posterior
nsample = 1000
trace = approx.sample(nsample, include_transformed = True)

data = trace_to_dataframe(trace, include_transformed=True, 
                          varnames=['ctau', 'c0', 
                                    'mtau', 'muA', 'muB', 
                                    'btau', 'beta',
                                    'dtau', 'delta', 
                                    'rtau', 'r'])
    
#get posterior samples of the predicted response value in the postreversal phase
gs = np.zeros((10000, 35, n_subs, 2))
for i in range(10):
    trace = approx.sample(nsample, include_transformed = True)
    gs[i*1000:(i+1)*1000] = trace['G'][:,last:]

post_like = np.exp(gs-gs.max(axis=-1)[:,:,:,np.newaxis])
post_like /= post_like.sum(axis = -1)[:,:,:,np.newaxis]

#get response outcomes in the post-reversal phase
obs = inp[np.newaxis, last:, :, 0]

#compute observation likelihood for each posterior sample
post_ol = post_like[:,:,:,0]**(1-obs)*post_like[:,:,:,1]**obs

#compute mean likelihood over all samples
mean_pl = post_ol.prod(axis = 1).mean(axis = 0)

#get posterior predictive log model evidence
pplme = np.log(mean_pl)

#get mean probability for each subject of slecting stimuli 2
plike2 = post_like[:,:,:,1].mean(axis = 0)

store = pd.HDFStore('results.h5')
store['edhmm/trace'] = data
store['edhmm/pplme'] = pd.Series(pplme)
store['edhmm/plike2'] = pd.DataFrame(plike2)
store.close()