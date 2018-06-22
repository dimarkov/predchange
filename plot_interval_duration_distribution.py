from __future__ import division, print_function
from builtins import range
import numpy as np
from scipy.special import binom, gammaln
from scipy.optimize import minimize_scalar

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context = 'talk', style = 'white', color_codes=True)
sns.set_style('ticks')

def binomln(x1, x2):
    return gammaln(x1+1) - gammaln(x2+1) - gammaln(x1-x2+1)

def fun(x,x0):
    return (x*(d_max*x**(d_max+1) - (d_max+1)*\
               x**d_max + 1)/((1-x)*(1-x**(d_max+1))) - x0)**2


def neg_binomial(mu, sigma, d_max):
    p = 1-(mu-1)/sigma
    r = (mu-1)**2/(sigma-mu+1)
    k = np.arange(d_max)
    l = binomln(k+r-1, k) + k*np.log(p) + r*np.log(1-p)
    D = np.exp(l)
    
    return D/D.sum()

def evolve_switch_prob2(S, D, T):
    tm = np.einsum('ijk, lk-> iljk', S, D)
    x = np.zeros(T)
    joint = np.einsum('ijkl, l -> ijk', tm, D[:,0])
    for t in range(T):
        x[t] = joint[1,:,0].sum()
        joint = np.einsum('ijkl, klm -> ijm', tm, joint)
    return x

def evolve_switch_prob(S, D, T):
    tm = np.einsum('ijk, lk-> iljk', S, D)
    x = np.zeros(T-1)
    joint = np.einsum('ijkl, l -> ijk', tm, D[:,0])[:,:,0]
    for t in range(T-1):
        if t == 0:
            x[t] = joint.sum(axis = 1)[1]
        else:
            prob = joint.sum(axis = 1)
            x[t] = prob[1,0]/prob[:,0].sum()
            joint = joint.sum(axis = -1)
        
        joint = np.einsum('ijkl, kl -> ijk', tm, joint)
    return x

#def evolve_switch_prob2(D, T):
#    S = np.zeros(2)
#    S[0] = 1
#
#    stm = np.zeros((2,2,len(D)))
#    stm[:,:,1:] = np.eye(2)[:,:,np.newaxis]
#    stm[:,:,0] = np.ones((2,2)) - np.eye(2)
#    
#    dtm = np.zeros((len(D), len(D)))
#    dtm[:,0] = D
#    for d in range(1,len(D)):
#        dtm[d-1, d] = 1
#    
#    
#    x = np.zeros(T)
#    prior = D.copy()
#    p = stm.copy()
#    for t in range(T):
#        l = len(p.shape)
#        lista = list(range(l))
#        pc = np.einsum(p, lista, 
#                      dtm, [l-1, l], 
#                      list(range(l-1))+[l])
#        
#        p = np.einsum(pc, lista,
#                      stm, [l-2, l, l-1], 
#                      list(range(l-1))+[l, l-1])
#        
#    return p
        
    

d_max = 200 #maximal interval duration

ds = 2
R = (np.ones((ds,ds)) - np.eye(ds))/(ds-1)
S = np.zeros((ds,ds,d_max))
S[:,:,0] = R
S[:,:,1:] = np.eye(ds)[:,:,np.newaxis]

D = np.zeros((d_max, d_max))
for d in range(1,d_max):
    D[d-1, d] = 1 

fig1, ax1 = plt.subplots(figsize = (10,5))
fig2, ax2 = plt.subplots(figsize = (10,5))

ax1.vlines(20, 0 ,.15, color = 'k', linestyle='--', label = r'$\mu$')
ax2.vlines(20, 0 ,.09, color = 'k', linestyle='--', label = r'$\mu$')

delta_df = pd.DataFrame()
mu = 20
labels = [r'$\sigma = \mu$', r'$\sigma = 10\mu$', r'$\sigma = (\mu-1)\mu$']
for i, x in enumerate([1, 10, 19]):
    sigma = mu*x
    p0 = neg_binomial(mu, sigma, d_max)
    D[:,0] = p0
    ax2.bar(np.arange(d_max)+1, p0, alpha = 0.5, label = labels[i])
    T = 100
    ax1.plot(np.arange(2, T+1, 1), evolve_switch_prob(S, D, T), 
             'd', alpha = 0.8, label = labels[i])


ax1.legend()
ax1.set_xlim([1,80])
ax2.legend()
ax2.set_xlim([1,40])
ax2.set_xlabel('$d$')
ax2.set_ylabel('$p_0(d)$')

ax1.set_ylabel(r'$\delta_{\tau}$')
ax1.set_xlabel(r'$\tau$')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False) 
ax2.spines['right'].set_visible(False)

fig1.savefig('Fig2.pdf', bbox_inches = 'tight', transparent = True)
fig2.savefig('Fig1.pdf', bbox_inches = 'tight', transparent = True)