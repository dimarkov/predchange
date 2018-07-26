#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This scripts contains pymc3 implementaiton of the ED-HMM and DU-RW models.
We used this funcitons for esitmating posterior distribution over 
free model parameters, either from the behavioral or simulated data.
"""

from pymc3.distributions.timeseries import scan

from theano.tensor import log
import theano.tensor as tt

from pymc3 import Model, fit
from pymc3 import Deterministic, HalfCauchy, Categorical

from theano_models import durw_model, edhmm_model

def durw_fit(inp, nans, n_subs, last):
    # inp - array containing responses, outcomes, and a switch variable witch turns off update in the presence of nans
    # nans - bool array pointing towards locations of nan responses and outcomes
    # n_subs - int value, total number of subjects (each subjects is fited to a different parameter value)
    # last - int value, negative value denoting number of last trials to exclude from parameter estimation
    #        e.g. setting last = -35 excludes the last 35 trials from parameter estimation.
    
    #define the hierarchical parametric model for DU-RW
    with Model() as durw:

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
        (Q, _) = scan(durw_model, sequences = [inp],
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
        Categorical('obs', p = p, observed = inp[:last,:,0][~nans[:last]])

    #fit the model    
    with durw:
        approx = fit(method = 'advi', n = 50000, progressbar = True)
    
    return approx

def edhmm_fit(inp, nans, n_subs, last):
    # inp - array containing responses, outcomes, and a switch variable witch turns off update in the presence of nans
    # nans - bool array pointing towards locations of nan responses and outcomes
    # n_subs - int value, total number of subjects (each subjects is fited to a different parameter value)
    # last - int value, negative value denoting number of last trials to exclude from parameter estimation
    #        e.g. setting last = -35 excludes the last 35 trials from parameter estimation.
    
    #define the hierarchical parametric model for ED-HMM
    #define the hierarchical parametric model
    d_max = 200 #maximal value for state duration
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
        Categorical('obs', p = p, observed = responses)
    
    #fit the model    
    with edhmm:
        approx = fit(method = 'advi', n = 50000, progressbar = True)
        
    return approx