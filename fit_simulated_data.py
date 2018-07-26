#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit ED-HMM and DU-RW model to simulated data
@author: dimitrije
"""

from pymc3_models import edhmm_fit, durw_fit

import numpy as np
import pandas as pd

from stats import sample_from_posterior

behavior = np.load('simulated_behavior.npy')
T, n_subs = behavior.shape[:2]
last = -35

inp = np.ones((T, n_subs,3), dtype = int)
inp[:,:,:2] = behavior.astype(int)

#define a dummy nans vector, as we have no failed responses in simulations
nans = np.zeros((T, n_subs), dtype = bool)

#fit models to the simulated data
approx_durw = durw_fit(inp, nans, n_subs, last)

approx_edhmm = edhmm_fit(inp, nans, n_subs, last)

#sample from posterior and save the results
pars = ['c0', 'beta']

varnames = pars + ['kappa', 'alpha', 'V0']
sample, pplme, plike2 = sample_from_posterior(approx_durw, 
                                              varnames, 
                                              n_subs,
                                              inp[last:,:,0],
                                              last)
store = pd.HDFStore('simulation_fits.h5')
store['durw/trace'] = sample
store['durw/pplme'] = pd.Series(pplme)
store['durw/plike2'] = pd.DataFrame(plike2)
store.close()

varnames = pars + ['delta', 'r', 'muA', 'muB']
sample, pplme, plike2 = sample_from_posterior(approx_edhmm, 
                                              varnames, 
                                              n_subs,
                                              inp[last:,:,0], 
                                              last)
store = pd.HDFStore('simulation_fits.h5')
store['edhmm/trace'] = sample
store['edhmm/pplme'] = pd.Series(pplme)
store['edhmm/plike2'] = pd.DataFrame(plike2)
store.close()

    
