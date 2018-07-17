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
from pymc3 import Categorical, HalfCauchy, Deterministic, InverseGamma

from theano_models import edhmm_model, durw_model

import numpy as np
import pandas as pd

def sample_from_posterior(approx, varnames, n_subs, responses):
    #sample from posterior
    nsample = 1000
    trace = approx.sample(nsample, include_transformed = True)
    
    sample = trace_to_dataframe(trace, include_transformed=True, 
                              varnames=varnames)
        
    #get posterior samples of the predicted response value in the postreversal phase
    gs = np.zeros((10000, 35, n_subs, 2))
    for i in range(10):
        trace = approx.sample(nsample, include_transformed = False)
        gs[i*1000:(i+1)*1000] = trace['G'][:,last:]
    
    post_like = np.exp(gs-gs.max(axis=-1)[:,:,:,None])
    post_like /= post_like.sum(axis = -1)[:,:,:,None]
    
    #measured responses in the post-reversal phase
    res = responses[None,:]
    
    #compute observation likelihood for each posterior sample
    post_ol = post_like[:,:,:,0]**(1-res)*post_like[:,:,:,1]**res
    
    #get posterior predictive log model evidence
    pplme = np.log(post_ol.prod(axis = 1).mean(axis = 0))
    
    #get per subject mean probability of selecting stimuli 2
    plike2 = post_like[:,:,:,1].mean(axis = 0)
    
    return sample, pplme, plike2 

behavior = np.load('simulated_behavior.npy')
d_max = 100
T, n_subs = behavior.shape[:2]
last = -35

inp = np.ones((T, n_subs,3), dtype = int)
inp[:,:,:2] = behavior.astype(int)

#observed responses in the first two phases
responses = inp[:last,:,0].reshape(-1)

#define the hierarchical parametric model for the edhmm agent
with Model() as edhmm:
    d = tt.arange(d_max) #vector of possible duration values from zero to d_max
    d = tt.tile(d, (n_subs,1))
    P = tt.ones((2,2)) - tt.eye(2) #permutation matrix
    
    #set prior state probability
    theta0 = tt.ones(n_subs)/2  
    
    #set hierarchical prior for delta parameter of prior beliefs p_0(d)
    dtau = HalfCauchy('dtau', beta = 1)
    dloc = HalfCauchy('dloc', beta = dtau, shape = (n_subs,))
    delta = Deterministic('delta', 1/(1+(dloc)**2)) 
    
    #set hierarchical prior for r parameter of prior beleifs p_0(d)
    rtau = HalfCauchy('rtau', beta = 1)
    rloc = HalfCauchy('rloc', beta = rtau, shape = (n_subs,))
    r = Deterministic('r', 1+rloc)
    
    #compute prior beliefs over state durations for given r and delta
    binomln = tt.gammaln(d+r[:,None]) - tt.gammaln(d+1) - tt.gammaln(r[:,None])
    pd0 = tt.nnet.softmax(binomln + d*log(1-delta[:,None]) + r[:,None]*log(delta[:,None]))
    
    #set joint probability distribution
    joint0 = tt.stack([theta0[:,None]*pd0, (1-theta0)[:,None]*pd0]).dimshuffle(1,0,2)
    
    #set hierarchical priors for response noises   
    btau = HalfCauchy('btau', beta = 1)
    bloc = HalfCauchy('bloc', beta = btau, shape=(n_subs,))
    beta = Deterministic('beta', 1/bloc)
    
    #set hierarchical priors for initial inital beliefs about reward probability
#    mtau = HalfCauchy('mtau', beta = 4)
#    mloc = HalfCauchy('mloc', beta = mtau, shape = (n_subs,2))
#    muA = Deterministic('muA', mloc[:,0]/(1+mloc[:,0])) 
#    muB = Deterministic('muB', 1/(1+mloc[:,1])) 
#    init = tt.stacklists([[10*muA, 10*(1-muA)], \
#                          [10*muB, 10*(1-muB)]]).dimshuffle(2,0,1)
    
    init = tt.stacklists([[tt.ones(n_subs)*8, tt.ones(n_subs)*2], \
                          [tt.ones(n_subs)*2, tt.ones(n_subs)*8]]).dimshuffle(2,0,1)
    
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
    mu = a/(a+b)
    theta = tt.concatenate([theta0[None, :], post[1][:-1].sum(axis=-1)[:,:,0]])[:,:,None]
    
    #compute choice dependend expected reward probability
    mean = theta*mu + (1-theta)*mu.dot(P)
    
    #compute expected utility
    U = 2*mean-1

    #set hierarchical prior for response biases
#    ctau = HalfCauchy('ctau', beta = 1)
#    cloc = HalfCauchy('cloc', beta = 1, shape = (n_subs,))
    c0 = tt.ones(n_subs)/2 #Deterministic('c0', 1/(1+(cloc)**2))

    #compute response noise and response bias modulated expected free energy
    G = Deterministic('G', beta[None,:, None]*U + log([c0, 1-c0]).T[None, :, :])
    
    #compute response probability for the prereversal and the reversal phase of the experiment
    p = Deterministic('p', tt.nnet.softmax(tt.reshape(G[:last],(n_subs*(T+last),2),ndim = 2)))
    
    #set observation likelihood of responses
    observed = Categorical('obs', p = p, observed = responses)
    
#define the hierarchical parametric model for DU-RW agent
du_inp = inp.copy()
du_inp[:,:,1]=2*du_inp[:,:,1]-1

with Model() as durw:

    #set hierarchical priors for learning rates
    atau = HalfCauchy('atau', beta = 1)
    aloc = HalfCauchy('aloc', beta = atau, shape = (n_subs,))
    alpha = Deterministic('alpha', 1/(1+(aloc)**2)) 
    
    #set hierarchical priors for coupling strengths
    ktau = HalfCauchy('ktau', beta = 1)
    kloc = HalfCauchy('kloc', beta = ktau, shape = (n_subs,))
    kappa = Deterministic('kappa', 1/(1+(kloc)**2))

    #set hierarchical priors for response noises   
    btau = HalfCauchy('btau', beta = 1)
    bloc = HalfCauchy('bloc', beta = btau, shape=(n_subs,))
    beta = Deterministic('beta', 1/bloc)
    
    #set hierarchical priors for initial choice value
    #mtau = HalfCauchy('mtau', beta = 1)
#    mlocA = HalfCauchy('mlocA', beta = 1, shape = (n_subs,))
#    mlocB = HalfCauchy('mlocB', beta = 1, shape = (n_subs,))
#    muA = Deterministic('muA', 1/(1+(mlocA)**2)) 
#    muB = Deterministic('muB', 1/(1+(mlocB)**2)) 
    V0 = tt.zeros((n_subs,2))#tt.stacklists([2*muA -1, 2*muB -1]).T
    
    #compute the choice values
    (Q, _) = scan(durw_model, sequences = [du_inp],
                 outputs_info = V0,
                 non_sequences = [alpha, kappa, range(n_subs)],
                 name = 'rw')

    V0 = V0[None,:,:]
    V = tt.concatenate([V0, Q[:-1]])
    
    #set hierarchical prior for response biases
#    ctau = HalfCauchy('ctau', beta = 1)
#    cloc = HalfCauchy('cloc', beta = 1, shape = (n_subs,))
    c0 = tt.ones(n_subs)/2 #Deterministic('c0', 1/(1+(cloc)**2))

    #compute response noise and response bias modulated response values
    G = Deterministic('G', beta[None,:, None]*V + log([c0, 1-c0]).T[None, :, :])
    
    #compute response probability for the prereversal and the reversal phase of the experiment
    p = tt.nnet.softmax(tt.reshape(G[:last],(n_subs*(T+last),2),ndim = 2))

    #set observation likelihood of responses
    observed = Categorical('obs', p = p, observed = responses)

pars = ['beta']#, 'muA', 'muB']

#fit edhmm model    
with edhmm:
    approx = fit(method = 'advi', n = 50000)

varnames = pars + ['delta', 'r']
sample, pplme, plike2 = sample_from_posterior(approx, 
                                              varnames, 
                                              n_subs,
                                              inp[last:,:,0])
store = pd.HDFStore('simulation_fits.h5')
store['edhmm/trace'] = sample
store['edhmm/pplme'] = pd.Series(pplme)
store['edhmm/plike2'] = pd.DataFrame(plike2)
store.close()

#fit durw model
with durw:
     approx = fit(method = 'advi', n = 50000)
     
varnames = pars + ['kappa', 'alpha']
sample, pplme, plike2 = sample_from_posterior(approx, 
                                              varnames, 
                                              n_subs,
                                              inp[last:,:,0])
store = pd.HDFStore('simulation_fits.h5')
store['durw/trace'] = sample
store['durw/pplme'] = pd.Series(pplme)
store['durw/plike2'] = pd.DataFrame(plike2)
store.close()