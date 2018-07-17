"""
Collection of helper functions.
"""
from __future__ import division
#from builtins import range

import numpy as np
from scipy.special import gammaln

from scipy.stats import entropy

from pymc3 import trace_to_dataframe

def binomln(x1, x2):
    return gammaln(x1+1) - gammaln(x2+1) - gammaln(x1-x2+1)

def neg_binomial(par1, par2, d, transformed = True):
    if transformed:
        dlt = (par1-1)/par2
        r = (par1-1)**2/(par2-par1+1)
    else:
        dlt = par1
        r = par2
    
    l = binomln(d+r-1, d) + d*np.log(1-dlt) + r*np.log(dlt)
    D = np.exp(l-l.max())
    
    return D/D.sum()

def evolve_change_prob(S, D, T):
    # joint transition matrix
    tm = np.einsum('ijk, lk-> iljk', S, D)
    
    #change probability
    dlt_tau = np.zeros(T-1)
    
    #joint_probability if the previous state was non reversal 
    #and the previous state duration follow the prior p0(d)
    joint = np.einsum('ijkl, l -> ijk', tm, D[:,0])[:,:,0]
    
    #estimate change probability delta_tau at a future trial tau
    for tau in range(T-1):
        marg_prob = joint.sum(axis = 1)
        if tau == 0:
            dlt_tau[tau] = marg_prob[1]
        else:
            dlt_tau[tau] = marg_prob[1,0]/marg_prob[:,0].sum()
            joint = joint.sum(axis=-1)
        
        joint = np.einsum('ijkl, kl -> ijk', tm, joint)
    
    return dlt_tau 

def kl_div(P, Q):
    
    return entropy(P, Q)

def sample_from_posterior(approx, varnames, n_subs, responses, last):
    #sample from posterior
    nsample = 1000
    trace = approx.sample(nsample, include_transformed = True)
    
    sample = trace_to_dataframe(trace, include_transformed=True, 
                              varnames=varnames)
        
    #get posterior samples of the predicted response value in the postreversal phase
    gs = np.zeros((10000, -last, n_subs, 2))
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