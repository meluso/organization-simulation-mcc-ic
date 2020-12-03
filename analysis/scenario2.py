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
    m, se = np.mean(a,axis=axis), scipy.stats.sem(a,axis=axis)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


#Specify plot to generate
plot_num = 2

# Define constants
n_steps = 100
x_values = np.arange(n_steps)

if plot_num == 1:

    # Import results
    loc = '../data/culture_sim_exec001/results.npy'
    mcc, inc, prf, dem, lvl = dm.load_exec001_results(loc)

    # Create the plot
    plt.figure(figsize=(7,4))
    plt.suptitle("Avg. of Runs w/ Beta Culture Distribution")

    # Plot culture results
    plt.subplot(1,2,1)
    plt.plot(x_values,100*mcc[1,:],label='MCC=0.1')
    plt.plot(x_values,100*mcc[2,:],label='MCC=0.2')
    plt.plot(x_values,100*mcc[3,:],label='MCC=0.3')
    plt.plot(x_values,100*mcc[4,:],label='MCC=0.4')
    plt.plot(x_values,100*mcc[5,:],label='MCC=0.5')
    plt.plot(x_values,100*mcc[6,:],label='MCC=0.6')
    plt.plot(x_values,100*mcc[7,:],label='MCC=0.7')
    plt.plot(x_values,100*mcc[8,:],label='MCC=0.8')
    plt.plot(x_values,100*mcc[9,:],label='MCC=0.9')
    plt.xlabel('Turns')
    plt.ylabel('Contest-Orientation Prevalence (%)')
    plt.ylim(0, 100)

    # Plot performance results
    plt.subplot(1,2,2)
    plt.plot(x_values,prf[1,:],label='MCC=0.1')
    plt.plot(x_values,prf[2,:],label='MCC=0.2')
    plt.plot(x_values,prf[3,:],label='MCC=0.3')
    plt.plot(x_values,prf[4,:],label='MCC=0.4')
    plt.plot(x_values,prf[5,:],label='MCC=0.5')
    plt.plot(x_values,prf[6,:],label='MCC=0.6')
    plt.plot(x_values,prf[7,:],label='MCC=0.7')
    plt.plot(x_values,prf[8,:],label='MCC=0.8')
    plt.plot(x_values,prf[9,:],label='MCC=0.9')
    plt.xlabel('Turns')
    plt.ylabel('Organization Performance')
    plt.ylim(0, 1)
    plt.legend(loc='upper left',bbox_to_anchor=(1.05, 1),borderaxespad=0.)

    # Show figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

elif plot_num == 2:

    # Import results
    loc = '../data/culture_sim_exec002/results.npy'
    mcc, prf, lvl = dm.load_exec002_results(loc)
    cases = dm.cases_exec002()

    # Calculate culture change
    mcc_delta = mcc[1:,:,-1]-mcc[1:,:,0]

    # Calculate values to plot
    mcc_start = np.mean(mcc[1:,:,0],axis=1)
    mcc_mean, mcc_err = mean_confidence_interval(mcc_delta,axis=1)

    # Create the plot
    plt.figure(figsize=(7,4))
    plt.suptitle("Avg. of Runs w/ Beta Culture Distribution")

    # Plot absolute change
    plt.subplot(1,2,1)
    plt.plot(mcc_start,mcc_mean)
    plt.xlabel('Starting Contest-Orientation')
    plt.ylabel('Change in Contest-Orientation (Absolute)')
    plt.grid(True)

    # Plot relative change
    plt.subplot(1,2,2)
    plt.plot(mcc_start,100*np.divide(mcc_mean,mcc_start))
    plt.xlabel('Starting Contest-Orientation')
    plt.ylabel('Change in Contest-Orientation (%)')
    plt.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

elif plot_num == 3:

    # Import results
    loc = '../data/culture_sim_exec002/results.npy'
    mcc, prf, lvl = dm.load_exec002_results(loc)

    # Calculate lvl change
    lvl_delta = lvl[1:,:,-1,:]-lvl[1:,:,0,:]

    # Calculate values to plot
    lvl_start = np.mean(lvl[1:,:,0,:],axis=1)
    lvl_mean, lvl_err = mean_confidence_interval(lvl_delta,axis=1)

    # Create the plot
    fig = plt.figure(figsize=(7,4))
    plt.suptitle("Avg. of Runs w/ Beta Culture Distribution")

    # Plot absolute change
    ax1 = plt.subplot(1,2,1)
    plt.plot(lvl_start[:,0],lvl_mean[:,0],label='Level 1')
    plt.plot(lvl_start[:,0],lvl_mean[:,1],label='Level 2')
    plt.plot(lvl_start[:,0],lvl_mean[:,2],label='Level 3')
    plt.plot(lvl_start[:,0],lvl_mean[:,3],label='Level 4')
    plt.plot(lvl_start[:,0],lvl_mean[:,4],label='Level 5')
    plt.xlabel('Starting Contest-Orientation')
    plt.ylabel('Change in Contest-Orientation (Absolute)')
    plt.grid(True)

    # Plot relative change
    ax2 = plt.subplot(1,2,2)
    plt.plot(lvl_start[:,0],100*np.divide(lvl_mean[:,0],lvl_start[:,0]),label='Level 1')
    plt.plot(lvl_start[:,1],100*np.divide(lvl_mean[:,1],lvl_start[:,1]),label='Level 2')
    plt.plot(lvl_start[:,2],100*np.divide(lvl_mean[:,2],lvl_start[:,2]),label='Level 3')
    plt.plot(lvl_start[:,3],100*np.divide(lvl_mean[:,3],lvl_start[:,3]),label='Level 4')
    plt.plot(lvl_start[:,4],100*np.divide(lvl_mean[:,4],lvl_start[:,4]),label='Level 5')
    plt.xlabel('Starting Contest-Orientation')
    plt.ylabel('Change in Contest-Orientation (%)')
    plt.grid(True)

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.025),
               borderaxespad=0.,ncol=5)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.show()