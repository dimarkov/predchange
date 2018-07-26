import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set(context = 'talk', style = 'white', color_codes=True)
sns.set_palette([ '#e41a1c', '#377eb8', '#4daf4a', '#984ea3'])

T = 160 #number of trials for each experimental block
                 
mean_response = np.load('mean_responses_exp.npy')
labels = ['IRI', 'RRI', 'SU-RW', 'DU-RW']
             
gs = gridspec.GridSpec(nrows=1, ncols=3)

fig = plt.figure(figsize = (12,4))
fig.subplots_adjust(wspace = .5)

ax1 = fig.add_subplot(gs[0, :2])
ax2 = fig.add_subplot(gs[0, -1])

ax1.plot(np.arange(1,T+1), mean_response)
sns.despine(fig=fig, ax = ax1)

ax1.legend(labels, title = 'agent')
ax1.set_xlim([1, 160])
ax1.set_ylim([0,1])
ax1.vlines([55, 70, 90, 105, 125], 0, 1, 'k', linestyle = '--')
ax1.set_ylabel(r'$Pr($choice = correct$)$')
ax1.set_xlabel('trial')


class_number = np.array([9, 16,  1,  1]); #the values are obtained from model_comparison.py
class_prob = np.vstack([1-class_number/25, class_number/25])

sns.heatmap(data = class_prob.T, ax = ax2, annot = True, vmax = 1, vmin = 0)
ax2.set_yticklabels(['DU-RW', 'SU-RW', 'RRI', 'IRI'], va = 'center')
ax2.set_xticklabels(['DU-RW', 'ED-HMM'])

ax2.set_ylabel('true model')
ax2.set_xlabel('inferred model')

ax1.text(-15, 1, 'A')
ax2.text(-.5, 4, 'B')

fig.savefig('Fig8.pdf', bbox_inches='tight', transparent = True)