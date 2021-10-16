# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:59:35 2020

@author: John Meluso
"""

import numpy as np
import matplotlib.pyplot as plt
import data_manager as dm
import scipy.stats
from matplotlib.colors import LinearSegmentedColormap

# CASE, RUNS, STEPS[, LEVELS]


def mean_confidence_interval(data, axis, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a,axis=axis), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


# Create colors
cbin = {'mcc': '#F47D20', 'ic': '#66AC47'}
crng_cultures = ['#66AC47','#F47D20']
crng_levels = ['#000000','#1375AF','#90D4ED']
crng_perfs = ['#000000','#007155']
cmap = LinearSegmentedColormap.from_list("mycmap", crng_levels)
list_culture = cmap(np.linspace(0,1,9))
list_level = cmap(np.linspace(0,1,5))

# Import results
loc = '../data/culture_sim_exec002/results.npy'
mcc, prf, lvl = dm.load_exec002_results(loc)

# Calculate values to plot
mcc_mean, mcc_err = mean_confidence_interval(mcc[0,:,:],axis=0)
prf_mean, prf_err = mean_confidence_interval(prf[0,:,:],axis=0)
lvl_mean, lvl_err = mean_confidence_interval(lvl[0,:,:,:],axis=0)

# Define constants
n_steps = 100
x_values = np.arange(n_steps)

# Import levels
levels = dm.generate_levels()

# Create the plot
plt.figure(figsize=(7.5,5),dpi=300)
#plt.suptitle("Avg. of Runs w/ Uniform Culture Distribution")

# Plot culture results
plt.subplot(1,3,1)
plt.plot(x_values,mcc_mean,label='Contest-Orientation',color=cbin['mcc'])
plt.plot(x_values,(1-mcc_mean),label='Inclusiveness',color=cbin['ic'])
plt.xlabel('Turns')
plt.ylabel('Attribute Prevalence')
plt.ylim(0, 1)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.2),borderaxespad=0.)

# Plot performance results
plt.subplot(1,3,2)
labels = ['Level 1','Level 2','Level 3','Level 4','Level 5']
for ii, label in enumerate(labels):
    plt.plot(x_values,lvl_mean[:,ii],label=label,color=list_level[ii])
plt.xlabel('Turns')
plt.ylabel('Contest-Orientation Prevalence')
plt.ylim(0, 1)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.2),borderaxespad=0.)

# Plot performance results
plt.subplot(1,3,3)
plt.plot(x_values,prf_mean,label='Performance',color=crng_perfs[1])
plt.xlabel('Turns')
plt.ylabel('Organization Performance')
plt.ylim(0, 1)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.2),borderaxespad=0.)

# Show figure
plt.tight_layout(rect=[0, 0, 1, 0.95])
loc = 'C:/Users/Juango the Blue/Documents/2020-2022 (Vermont)/Conferences/Networks 2021'
plt.savefig(loc + '/Uniform Means.svg')
plt.show()