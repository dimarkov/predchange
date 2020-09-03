#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit the ehavioral data with ED-HMM and DU-RW models.
"""

from os import walk

import numpy as np
import pandas as pd

from stats import sample_from_posterior

from pymc3_models import durw_fit, edhmm_fit

#get the file names
f = []
for (dirpath, dirnames, filenames) in walk('data'):
    f.extend(filenames)
    break
f = np.sort(f)[1:]

#set parameters
n_subs = len(f) #number of subjects

T = 160 #total number of trials
last = -35 # set the time point for the post-reversal phase

#input variables for inference
inp = np.zeros((T, n_subs, 3), dtype = np.int64)

#falied trials
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
    
approx_durw = durw_fit(inp, nans, n_subs, last)

approx_edhmm = edhmm_fit(inp, nans, n_subs, last)

#sample from posterior and estimate posterior predictive log-model evidence
# varnames = ['c0', 'beta', 'alpha', 'kappa', 'V0']
# sample, pplme, plike2 = sample_from_posterior(approx_durw, 
#                                               varnames, 
#                                               n_subs,
#                                               inp[last:,:,0], 
#                                               last)

# store = pd.HDFStore('behaivor_fits.h5')
# store['durw/trace'] = sample
# store['durw/pplme'] = pd.Series(pplme)
# store['durw/plike2'] = pd.DataFrame(plike2)
# store.close()


# varnames = ['c0', 'beta', 'delta', 'r', 'muA', 'muB']
# sample, pplme, plike2 = sample_from_posterior(approx_edhmm, 
#                                               varnames, 
#                                               n_subs,
#                                               inp[last:,:,0], 
#                                               last)

# store = pd.HDFStore('behavior_fits.h5')
# store['edhmm/trace'] = sample
# store['edhmm/pplme'] = pd.Series(pplme)
# store['edhmm/plike2'] = pd.DataFrame(plike2)
# store.close()