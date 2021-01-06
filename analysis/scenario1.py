# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:59:35 2020

@author: John Meluso
"""

import numpy as np
import matplotlib.pyplot as plt
import data_manager as dm
import scipy.stats

# CASE, RUNS, STEPS[, LEVELS]


def mean_confidence_interval(data, axis, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a,axis=axis), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


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
plt.plot(x_values,mcc_mean,label='Contest-Orientation')
plt.plot(x_values,(1-mcc_mean),label='Inclusiveness')
plt.xlabel('Turns')
plt.ylabel('Attribute Prevalence')
plt.ylim(0, 1)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.2),borderaxespad=0.)

# Plot performance results
plt.subplot(1,3,2)
plt.plot(x_values,lvl_mean[:,0],label='Level 1')
plt.plot(x_values,lvl_mean[:,1],label='Level 2')
plt.plot(x_values,lvl_mean[:,2],label='Level 3')
plt.plot(x_values,lvl_mean[:,3],label='Level 4')
plt.plot(x_values,lvl_mean[:,4],label='Level 5')
plt.xlabel('Turns')
plt.ylabel('Contest-Orientation Prevalence')
plt.ylim(0, 1)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.2),borderaxespad=0.)

# Plot performance results
plt.subplot(1,3,3)
plt.plot(x_values,prf_mean,label='Performance')
plt.xlabel('Turns')
plt.ylabel('Organization Performance')
plt.ylim(0, 1)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.2),borderaxespad=0.)

# Show figure
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()