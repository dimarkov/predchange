"""
Collection of helping functions.
"""
from __future__ import division
from builtins import range

import numpy as np
from scipy.special import gammaln

def binomln(x1, x2):
    return gammaln(x1+1) - gammaln(x2+1) - gammaln(x1-x2+1)

def neg_binomial(mu, sigma, k):
    p = 1-(mu-1)/sigma
    r = (mu-1)**2/(sigma-mu+1)
    l = binomln(k+r-1, k) + k*np.log(p) + r*np.log(1-p)
    D = np.exp(l-l.max())
    
    return D/D.sum()