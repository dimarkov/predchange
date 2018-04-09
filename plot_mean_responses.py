#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from os import walk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context = 'talk', style = 'white', color_codes = True)
sns.set_style('ticks')


######load the data

f = []
for (dirpath, dirnames, filenames) in walk('data'):
    f.extend(filenames)
    break
f = np.sort(f)[1:]

n_subjects = 22

n_last = 35 #number of last trials
responses = np.zeros((n_subjects, n_last))
outcomes = np.zeros((n_subjects, n_last))

for i,file in enumerate(f[:n_subjects]):
    sub, _ = file.split(sep='.')
    sub = int(sub[-2:])
    data = pd.read_csv(dirpath + '/' + file)
    start = data['S'][0] #get the initial state
    
    #set reponses to 0,1 values and realign them as if 
    #all participants started in the same initial state
    if start == 1:
        data['A'] -= 1
    else:
        data['A'] = 2 - data['A']
    
    responses[i, :] = data['A'].values[-n_last:]
    outcomes[i,:] = data['R'].values[-n_last:]

from scipy.special import betaincinv
#du-rw group
subs1 = np.array([0,1,4,5,6,9,11,12,13,14,15,16,18,19,20,21])
nans1 = np.isnan(responses[subs1, :]).sum(axis = 0)
p1 = np.nan_to_num(responses[subs1, :]).sum(axis = 0)

#edhmm group
subs2 = np.array([2,3,7,8,10,17])
nans2 = np.isnan(responses[subs2, :]).sum(axis = 0)
p2 = np.nan_to_num(responses[subs2, :]).sum(axis = 0)

trials = np.arange(126, 161, 1)

#compute 90% Jeffreys interval
n1 = len(subs1)
u1 = betaincinv(p1+1/2, n1-nans1-p1 + 1/2, .95)
l1 = betaincinv(p1+1/2, n1-nans1-p1 + 1/2, .05)
n2 = len(subs2)
u2 = betaincinv(p2+1/2, n2-nans2-p2 + 1/2, .95)
l2 = betaincinv(p2+1/2, n2-nans2-p2 + 1/2, .05)

#compute the mean response
m1 = p1/(n1-nans1+1)
m2 = p2/(n2-nans2+1)

#make the figure
fig, ax = plt.subplots(1,2, figsize = (12,5), sharex = True)

ax[0].plot(trials, m1, label = 'DU-RW group', color = 'r');
ax[0].fill_between(trials, u1, l1, color = 'r', alpha = .2) 
ax[0].plot(trials, m2, color = 'b', label = 'ED-HMM group');
ax[0].fill_between(trials, u2, l2, color = 'b', alpha = .2)
ax[0].legend()


#compute mean responses from random asignment
diff = np.zeros((10000, 35))
subs = np.arange(22)
for i in range(10000):
    #sample random subjects
    s2 = np.random.choice(subs,size = 6,replace = False)
    s1 = np.setdiff1d(subs, s2)
    r1 = np.nan_to_num(responses[s1,:]).sum(axis = 0)
    nan1 = np.isnan(responses[s1,:]).sum(axis = 0)
    r2 = np.nan_to_num(responses[s2,:]).sum(axis = 0)
    nan2 = np.isnan(responses[s2,:]).sum(axis = 0)
    
    diff[i] = r1/(n1-nan1+1) - r2/(n2-nan2+1)

ax[1].fill_between(trials, 
                     np.percentile(diff, 95, axis = 0), 
                     np.percentile(diff, 5, axis = 0), 
                     alpha = 0.5, color = 'c')

ax[1].fill_between(trials, 
                     np.percentile(diff, 75, axis = 0), 
                     np.percentile(diff, 25, axis = 0), 
                     alpha = 0.5, color = 'c')
    
ax[1].plot(trials, m1 - m2, color = 'k')

ax[0].set_ylabel('average group response')
ax[0].set_xlabel('trial')
ax[1].set_ylabel('group response difference')
ax[1].set_xlabel('trial')

ax[0].text(-0.16,1,'A', transform = ax[0].transAxes)
ax[1].text(-0.2,1,'B',transform = ax[1].transAxes)
fig.subplots_adjust(wspace=.25)
fig.savefig('Fig8.pdf', bbox_inches = 'tight', transparent = True)