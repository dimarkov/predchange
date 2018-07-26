#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the mean predictive response probability over different groups of subjects.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context = 'talk', style = 'white', color_codes = True)
sns.set_style('ticks')


subs1 = np.array([0,1,4,5,6,9,11,12,13,14,15,16,18,19,20,21])
subs2 = np.array([2,3,7,8,10,17])

######load the posterior predictive probability of selecting stimuli 2
store = pd.HDFStore('behavior_fits.h5')
plikeDU = store['durw/plike2']
plikeED = store['edhmm/plike2']
store.close()


fig, ax = plt.subplots(1,2, figsize = (10,5),sharey = True, sharex = True)
#plot DU-RW group in red
sns.tsplot(plikeDU.values[:, subs1].T, ax = ax[0], color = 'r', ci = 90);
sns.tsplot(plikeED.values[:, subs1].T, ax = ax[1], color = 'r', ci = 90);

#plot ED-HMM group in blue
sns.tsplot(plikeDU.values[:, subs2].T, ax = ax[0], color = 'b', ci = 90);
sns.tsplot(plikeED.values[:, subs2].T, ax = ax[1], color = 'b', ci = 90);


ax[0].set_xlim([10,30])
ax[1].set_xlim([10,30])
ax[0].set_ylim([.3,1])   
ax[0].set_xlabel('trial')
ax[0].set_ylabel('choice probability')
ax[0].set_xticklabels(np.arange(136,157,5))
ax[0].set_title('DU-RW')
ax[1].set_title('ED-HMM')

fig.savefig('Fig11.pdf', bbox_inches = 'tight', transparent = True)