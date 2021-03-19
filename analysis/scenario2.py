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
plot_num = 8

# Define constants
n_steps = 100
x_values = np.arange(n_steps)

if plot_num == 1:

    # Import results
    loc = '../data/culture_sim_exec001/results.npy'
    mcc, inc, prf, dem, lvl = dm.load_exec001_results(loc)

    # Create the plot
    plt.figure(figsize=(7,4),dpi=300)
    #plt.suptitle("Avg. of Runs w/ Beta Culture Distribution")

    # Plot culture results
    plt.subplot(1,2,1)
    plt.plot(x_values,mcc[1,:],label='MCC=0.1')
    plt.plot(x_values,mcc[2,:],label='MCC=0.2')
    plt.plot(x_values,mcc[3,:],label='MCC=0.3')
    plt.plot(x_values,mcc[4,:],label='MCC=0.4')
    plt.plot(x_values,mcc[5,:],label='MCC=0.5')
    plt.plot(x_values,mcc[6,:],label='MCC=0.6')
    plt.plot(x_values,mcc[7,:],label='MCC=0.7')
    plt.plot(x_values,mcc[8,:],label='MCC=0.8')
    plt.plot(x_values,mcc[9,:],label='MCC=0.9')
    plt.xlabel('Turns')
    plt.ylabel('Contest-Orientation Prevalence')
    plt.ylim(0, 1)

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

    # Import results & zoomed results
    loc = '../data/culture_sim_exec002/results.npy'
    mcc, prf, lvl = dm.load_exec002_results(loc)
    loc = '../data/culture_sim_exec003/results.npy'
    mcc_z, prf_z, lvl_z = dm.load_exec003_results(loc)
    cases = dm.cases_exec002()

    # Calculate culture change
    mcc_delta = mcc[1:,:,-1]-mcc[1:,:,0]
    mcc_delta_z = mcc_z[1:,:,-1]-mcc_z[1:,:,0]

    # Calculate absolute values to plot
    mcc_start = np.mean(mcc[1:,:,0],axis=1)
    mcc_start_z = np.mean(mcc_z[1:,:,0],axis=1)
    mcc_mean, mcc_err = mean_confidence_interval(mcc_delta,axis=1)
    mcc_mean_z, mcc_err_z = mean_confidence_interval(mcc_delta_z,axis=1)

    # Calculate percent values to plot
    mcc_pct = 100*np.divide(mcc_mean,mcc_start)
    mcc_pct_z = 100*np.divide(mcc_mean_z,mcc_start_z)
    mcc_pct_err = 100*np.divide(mcc_err,mcc_start)
    mcc_pct_err_z = 100*np.divide(mcc_err_z,mcc_start_z)

    # Create the plot
    fig = plt.figure(figsize=(7,4),
                     dpi=300
                     )
    #plt.suptitle("Avg. of Runs w/ Beta Culture Distribution")

    # Plot absolute change
    ax1 = plt.subplot(1,2,1)
    div = 94
    plt.plot(mcc_start[:div+1],mcc_mean[:div+1],label='$\Delta C\leq$0')
    plt.plot(mcc_start[div:],mcc_mean[div:],label='$\Delta C>$0')
    plt.xlabel('Starting Contest-Orientation')
    plt.ylabel('Change in Contest-Orientation (Absolute)')
    plt.xlim(0,1)
    plt.ylim(-0.06,0.015)
    plt.axhline(y=0,color='gray',linewidth=0.5)
    #plt.grid()

    # Plot ax insert
    axins1 = ax1.inset_axes([0.39, 0.56, 0.35, 0.35])
    div = 18
    x1, x2, y1, y2 = 0.9, 1.00, -0.0025, 0.0025
    axins1.set_xlim(x1,x2)
    axins1.set_ylim(y1,y2)

    axins1.plot(mcc_start_z[:div+1],mcc_mean_z[:div+1],label='$\Delta C\leq$0')
    axins1.plot(mcc_start_z[div:],mcc_mean_z[div:],label='$\Delta C>$0')
    axins1.fill_between(mcc_start_z[:div+1],
                        mcc_mean_z[:div+1] - mcc_err_z[:div+1],
                        mcc_mean_z[:div+1] + mcc_err_z[:div+1],
                        color='C0',alpha=0.2)
    axins1.fill_between(mcc_start_z[div:],
                        mcc_mean_z[div:] - mcc_err_z[div:],
                        mcc_mean_z[div:] + mcc_err_z[div:],
                        color='C1',alpha=0.2)

    axins1.set_xticks((0.9,0.945,1.0))
    axins1.set_xticklabels((0.9,0.945,1.0))
    axins1.tick_params(labelsize=9)
    axins1.axhline(y=0,color='gray',linewidth=0.5)
    axins1.axvline(x=0.945,color='gray',linewidth=0.5)
    ax1.indicate_inset_zoom(axins1)

    # Plot relative change
    ax2 = plt.subplot(1,2,2)
    div = 94

    plt.plot(mcc_start[:div+1],mcc_pct[:div+1],label='$\Delta C\leq$0')
    plt.plot(mcc_start[div:],mcc_pct[div:],label='$\Delta C>$0')

    plt.xlabel('Starting Contest-Orientation')
    plt.ylabel('Change in Contest-Orientation (%)')
    plt.xlim(0,1)
    plt.ylim(-42,2)
    plt.axhline(y=0,color='gray',linewidth=0.5)

    # Plot ax insert
    div = 18
    axins2 = ax2.inset_axes([0.54, 0.15, 0.4, 0.4])
    x1, x2, y1, y2 = 0.9, 1.00, -0.25, 0.25
    axins2.set_xlim(x1,x2)
    axins2.set_ylim(y1,y2)

    axins2.plot(mcc_start_z[:div+1],mcc_pct_z[:div+1],label='$\Delta C\leq$0')
    axins2.plot(mcc_start_z[div:],mcc_pct_z[div:],label='$\Delta C>$0')
    axins2.fill_between(mcc_start_z[:div+1],
                        mcc_pct_z[:div+1] - mcc_pct_err_z[:div+1],
                        mcc_pct_z[:div+1] + mcc_pct_err_z[:div+1],
                        color='C0',alpha=0.2,label='$\Delta C\leq0$, 95% CI')
    axins2.fill_between(mcc_start_z[div:],
                        mcc_pct_z[div:] - mcc_pct_err_z[div:],
                        mcc_pct_z[div:] + mcc_pct_err_z[div:],
                        color='C1',alpha=0.2,label='$\Delta C>0$, 95% CI')

    axins2.set_xticks((0.9,0.945,1.0))
    axins2.set_xticklabels((0.9,0.945,1.0))
    axins2.tick_params(labelsize=9)
    axins2.axhline(y=0,color='gray',linewidth=0.5)
    axins2.axvline(x=0.945,color='gray',linewidth=0.5)
    ax2.indicate_inset_zoom(axins2)

    handles, labels = axins2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.025),
               borderaxespad=0.,ncol=5)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.show()

elif plot_num == 3:

    # Import results
    loc = '../data/culture_sim_exec002/results.npy'
    mcc, prf, lvl = dm.load_exec002_results(loc)
    loc = '../data/culture_sim_exec003/results.npy'
    mcc_z, prf_z, lvl_z = dm.load_exec003_results(loc)

    # Calculate lvl change
    lvl_delta = lvl[1:,:,-1,:]-lvl[1:,:,0,:]
    lvl_delta_z = lvl_z[1:,:,-1,:]-lvl_z[1:,:,0,:]

    # Calculate values to plot
    lvl_start = np.mean(lvl[1:,:,0,:],axis=1)
    lvl_mean, lvl_err = mean_confidence_interval(lvl_delta,axis=1)
    lvl_start_z = np.mean(lvl_z[1:,:,0,:],axis=1)
    lvl_mean_z, lvl_err_z = mean_confidence_interval(lvl_delta_z,axis=1)

    # Create the plot
    fig = plt.figure(figsize=(7,4),dpi=300)
    #plt.suptitle("Avg. of Runs w/ Beta Culture Distribution")

    # Plot absolute change
    ax1 = plt.subplot(1,2,1)
    plt.plot(lvl_start[:,0],lvl_mean[:,0],label='Level 1')
    plt.plot(lvl_start[:,0],lvl_mean[:,1],label='Level 2')
    plt.plot(lvl_start[:,0],lvl_mean[:,2],label='Level 3')
    plt.plot(lvl_start[:,0],lvl_mean[:,3],label='Level 4')
    plt.plot(lvl_start[:,0],lvl_mean[:,4],label='Level 5')
    plt.xlabel('Starting Contest-Orientation')
    plt.ylabel('Change in Contest-Orientation (Absolute)')
    plt.axhline(y=0,color='gray',linewidth=0.5)

    # Plot relative change
    ax2 = plt.subplot(1,2,2)
    plt.plot(lvl_start[:,0],100*np.divide(lvl_mean[:,0],lvl_start[:,0]),label='Level 1')
    plt.plot(lvl_start[:,1],100*np.divide(lvl_mean[:,1],lvl_start[:,1]),label='Level 2')
    plt.plot(lvl_start[:,2],100*np.divide(lvl_mean[:,2],lvl_start[:,2]),label='Level 3')
    plt.plot(lvl_start[:,3],100*np.divide(lvl_mean[:,3],lvl_start[:,3]),label='Level 4')
    plt.plot(lvl_start[:,4],100*np.divide(lvl_mean[:,4],lvl_start[:,4]),label='Level 5')
    plt.xlabel('Starting Contest-Orientation')
    plt.ylabel('Change in Contest-Orientation (%)')
    plt.axhline(y=0,color='gray',linewidth=0.5)

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.025),
               borderaxespad=0.,ncol=5)

    # Inset for plot 2
    if False:
        axins2 = ax2.inset_axes([0.62, 0.12, 0.3, 0.3])
        x1, x2, y1, y2 = 0.9, 1.00, -1, 1
        axins2.set_xlim(x1,x2)
        axins2.set_ylim(y1,y2)
        axins2.plot(lvl_start_z[:,0],100*np.divide(lvl_mean_z[:,0],lvl_start_z[:,0]),label='Level 1')
        axins2.plot(lvl_start_z[:,1],100*np.divide(lvl_mean_z[:,1],lvl_start_z[:,1]),label='Level 2')
        axins2.plot(lvl_start_z[:,2],100*np.divide(lvl_mean_z[:,2],lvl_start_z[:,2]),label='Level 3')
        axins2.plot(lvl_start_z[:,3],100*np.divide(lvl_mean_z[:,3],lvl_start_z[:,3]),label='Level 4')
        axins2.plot(lvl_start_z[:,4],100*np.divide(lvl_mean_z[:,4],lvl_start_z[:,4]),label='Level 5')
        axins2.set_xticks((0.9,0.945,1.0))
        axins2.set_xticklabels((0.9,0.945,1.0))
        axins2.tick_params(labelsize=9)
        ax2.indicate_inset_zoom(axins2)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.show()

elif plot_num == 4:

    # Import results
    loc = '../data/culture_sim_exec002/results.npy'
    mcc, prf, lvl = dm.load_exec002_results(loc)

    # Calculate values to plot
    lvl_mean, lvl_err = mean_confidence_interval(lvl[1:,:,:,:], axis=1)
    lvl_runs = np.mean(lvl[1:,:,:,:],axis=1)
    lvl_deri = lvl_runs[:,1:,:]-lvl_runs[:,:-1,:]

    # Create the plot
    fig = plt.figure(figsize=(5,10),
                     dpi=300
                     )
    ax = fig.add_subplot(111)    # The big subplot

    # Create labels across multiple plots
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False,
                   right=False)
    fig.text(0.08, 0.56, 'Contest-Orientation Prevalence',
             ha='center', va='center',
             rotation='vertical')

    # MCC = 0.1
    ax1 = plt.subplot(5,1,1)
    plt.plot(x_values,lvl_runs[3,:,0],label='Level 1')
    plt.plot(x_values,lvl_runs[3,:,1],label='Level 2')
    plt.plot(x_values,lvl_runs[3,:,2],label='Level 3')
    plt.plot(x_values,lvl_runs[3,:,3],label='Level 4')
    plt.plot(x_values,lvl_runs[3,:,4],label='Level 5')
    plt.gcf().text(1, 0.91, 'MCC=0.03', fontsize=11)

    # MCC = 0.3
    ax2 = plt.subplot(5,1,2)
    plt.plot(x_values,lvl_runs[20,:,0],label='Level 1')
    plt.plot(x_values,lvl_runs[20,:,1],label='Level 2')
    plt.plot(x_values,lvl_runs[20,:,2],label='Level 3')
    plt.plot(x_values,lvl_runs[20,:,3],label='Level 4')
    plt.plot(x_values,lvl_runs[20,:,4],label='Level 5')
    plt.gcf().text(1, 0.725, 'MCC=0.20', fontsize=11)

    # MCC = 0.5
    ax3 = plt.subplot(5,1,3)
    plt.plot(x_values,lvl_runs[50,:,0],label='Level 1')
    plt.plot(x_values,lvl_runs[50,:,1],label='Level 2')
    plt.plot(x_values,lvl_runs[50,:,2],label='Level 3')
    plt.plot(x_values,lvl_runs[50,:,3],label='Level 4')
    plt.plot(x_values,lvl_runs[50,:,4],label='Level 5')
    plt.gcf().text(1, 0.55, 'MCC=0.50', fontsize=11)

    # MCC = 0.7
    ax4 = plt.subplot(5,1,4)
    plt.plot(x_values,lvl_runs[80,:,0],label='Level 1')
    plt.plot(x_values,lvl_runs[80,:,1],label='Level 2')
    plt.plot(x_values,lvl_runs[80,:,2],label='Level 3')
    plt.plot(x_values,lvl_runs[80,:,3],label='Level 4')
    plt.plot(x_values,lvl_runs[80,:,4],label='Level 5')
    plt.gcf().text(1, 0.375, 'MCC=0.80', fontsize=11)

    # MCC = 0.9
    ax5 = plt.subplot(5,1,5)
    plt.plot(x_values,lvl_runs[97,:,0],label='Level 1')
    plt.plot(x_values,lvl_runs[97,:,1],label='Level 2')
    plt.plot(x_values,lvl_runs[97,:,2],label='Level 3')
    plt.plot(x_values,lvl_runs[97,:,3],label='Level 4')
    plt.plot(x_values,lvl_runs[97,:,4],label='Level 5')
    # plt.fill_between(x_values,
    #                  (lvl_mean[97,:,1]-lvl_err[97,:,1]),
    #                  (lvl_mean[97,:,1]+lvl_err[97,:,1]),
    #                  color='C1',alpha=0.2)
    plt.gcf().text(1, 0.20, 'MCC=0.97', fontsize=11)
    ax5.set_xlabel('Turns')

    # Show figure
    handles, labels = ax5.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.6, 0.025), borderaxespad=0., ncol=5)
    plt.tight_layout(rect=[0.1, 0.08, 1, 1])
    plt.show()

elif plot_num == 5:

    # Import results
    loc = '../data/culture_sim_exec003/results.npy'
    mcc, prf, lvl = dm.load_exec003_results(loc)
    cases = dm.cases_exec003()

    # Calculate culture change
    mcc_delta = mcc[1:,:,-1]-mcc[1:,:,0]

    # Calculate values to plot
    mcc_start = np.mean(mcc[1:,:,0],axis=1)
    mcc_mean, mcc_err = mean_confidence_interval(mcc_delta,axis=1)

    # Create the plot
    plt.figure(figsize=(7,4),dpi=300)
    #plt.suptitle("Avg. of Runs w/ Beta Culture Distribution")

    # Plot absolute change
    plt.subplot(1,2,1)
    div = 17
    plt.plot(mcc_start[:div+1],mcc_mean[:div+1],label='$<$0')
    plt.plot(mcc_start[div:],mcc_mean[div:],label='$\geq$0')
    plt.xlabel('Starting Contest-Orientation')
    plt.ylabel('Change in Contest-Orientation (Absolute)')
    plt.legend()

    # Plot relative change
    plt.subplot(1,2,2)
    mcc_pct = 100*np.divide(mcc_mean,mcc_start)
    plt.plot(mcc_start[:div+1],mcc_pct[:div+1],label='$<$0')
    plt.plot(mcc_start[div:],mcc_pct[div:],label='$\geq$0')
    plt.xlabel('Starting Contest-Orientation')
    plt.ylabel('Change in Contest-Orientation (%)')
    plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

elif plot_num == 6:

    # Import results
    loc = '../data/culture_sim_exec003/results.npy'
    mcc, prf, lvl = dm.load_exec003_results(loc)
    cases = dm.cases_exec003()

    # Calculate lvl change
    lvl_delta = lvl[1:,:,-1,:]-lvl[1:,:,0,:]

    # Calculate values to plot
    lvl_start = np.mean(lvl[1:,:,0,:],axis=1)
    lvl_mean, lvl_err = mean_confidence_interval(lvl_delta,axis=1)

    # Create the plot
    fig = plt.figure(figsize=(7,4),dpi=300)
    #plt.suptitle("Avg. of Runs w/ Beta Culture Distribution")

    # Plot absolute change
    ax1 = plt.subplot(1,2,1)
    plt.plot(lvl_start[:,0],lvl_mean[:,0],label='Level 1')
    plt.plot(lvl_start[:,0],lvl_mean[:,1],label='Level 2')
    plt.plot(lvl_start[:,0],lvl_mean[:,2],label='Level 3')
    plt.plot(lvl_start[:,0],lvl_mean[:,3],label='Level 4')
    plt.plot(lvl_start[:,0],lvl_mean[:,4],label='Level 5')
    plt.xlabel('Starting Contest-Orientation')
    plt.ylabel('Change in Contest-Orientation (Absolute)')
    plt.ylim(-0.0125, 0.01)

    # Plot relative change
    ax2 = plt.subplot(1,2,2)
    plt.plot(lvl_start[:,0],100*np.divide(lvl_mean[:,0],lvl_start[:,0]),label='Level 1')
    plt.plot(lvl_start[:,1],100*np.divide(lvl_mean[:,1],lvl_start[:,1]),label='Level 2')
    plt.plot(lvl_start[:,2],100*np.divide(lvl_mean[:,2],lvl_start[:,2]),label='Level 3')
    plt.plot(lvl_start[:,3],100*np.divide(lvl_mean[:,3],lvl_start[:,3]),label='Level 4')
    plt.plot(lvl_start[:,4],100*np.divide(lvl_mean[:,4],lvl_start[:,4]),label='Level 5')
    plt.xlabel('Starting Contest-Orientation')
    plt.ylabel('Change in Contest-Orientation (%)')
    plt.ylim(-2, 1)

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.025),
               borderaxespad=0.,ncol=5)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.show()

elif plot_num == 7:

    # Import results & zoomed results
    loc = '../data/culture_sim_exec002/results.npy'
    mcc, prf, lvl = dm.load_exec002_results(loc)
    cases = dm.cases_exec002()

    # Calculate values to plot
    mcc_start = np.mean(mcc[1:,:,0],axis=1)
    prf_delta = prf[1:,:,-1] - prf[1:,:,0]
    prf_dlt_mean, prf_dlt_err = mean_confidence_interval(prf_delta,axis=1)
    prf_start = prf[1:,:,0]
    prf_pct_mean, prf_pct_err \
        = mean_confidence_interval(100*np.divide(prf_delta,prf_start),axis=1)

    # Create the plot
    fig = plt.figure(figsize=(7,4),
                           dpi=300
                           )

    # Create absolute change graph
    ax1 = plt.subplot(1,2,1)
    div = 94
    ax1.plot(mcc_start[:div+1],prf_dlt_mean[:div+1],label='$\Delta C\leq$0')
    ax1.plot(mcc_start[div:],prf_dlt_mean[div:],label='$\Delta C>$0')
    ax1.set_xlabel('Starting Contest-Orientation')
    ax1.set_ylabel('Performance Change (Absolute)')
    ax1.axhline(y=0,color='gray',linewidth=0.5)
    ax1.set_xlim(0,1)

    # Create percent change graph
    ax2 = plt.subplot(1,2,2)
    div = 94
    ax2.plot(mcc_start[:div+1],prf_pct_mean[:div+1],label='$\Delta C\leq$0')
    ax2.plot(mcc_start[div:],prf_pct_mean[div:],label='$\Delta C>$0')
    ax2.set_xlabel('Starting Contest-Orientation')
    ax2.set_ylabel('Performance Change (%)')
    ax2.axhline(y=0,color='gray',linewidth=0.5)
    ax2.set_xlim(0,1)

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.025),
               borderaxespad=0.,ncol=5)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.show()

elif plot_num == 8:

    # Import results & zoomed results
    loc = '../data/culture_sim_exec002/results.npy'
    mcc, prf, lvl = dm.load_exec002_results(loc)
    loc = '../data/culture_sim_exec003/results.npy'
    mcc_z, prf_z, lvl_z = dm.load_exec003_results(loc)
    cases = dm.cases_exec002()

    # Calculate culture change
    mcc_delta = mcc[1:,:,-1]-mcc[1:,:,0]
    mcc_delta_z = mcc_z[1:,:,-1]-mcc_z[1:,:,0]

    # Calculate absolute values to plot
    mcc_start = np.mean(mcc[1:,:,0],axis=1)
    mcc_start_z = np.mean(mcc_z[1:,:,0],axis=1)
    mcc_mean, mcc_err = mean_confidence_interval(mcc_delta,axis=1)
    mcc_mean_z, mcc_err_z = mean_confidence_interval(mcc_delta_z,axis=1)

    # Calculate percent values to plot
    mcc_pct = 100*np.divide(mcc_mean,mcc_start)
    mcc_pct_z = 100*np.divide(mcc_mean_z,mcc_start_z)
    mcc_pct_err = 100*np.divide(mcc_err,mcc_start)
    mcc_pct_err_z = 100*np.divide(mcc_err_z,mcc_start_z)

    # Create the plot
    fig = plt.figure(figsize=(4,4),
                     dpi=300
                     )
    #plt.suptitle("Avg. of Runs w/ Beta Culture Distribution")

    # Plot absolute change
    ax1 = fig.gca()
    div = 94
    plt.plot(mcc_start[:div+1],mcc_mean[:div+1],label='$\Delta C\leq$0')
    plt.plot(mcc_start[div:],mcc_mean[div:],label='$\Delta C>$0')
    plt.xlabel('Starting Contest-Orientation')
    plt.ylabel('Change in Contest-Orientation (Absolute)')
    plt.xlim(0,1)
    plt.ylim(-0.06,0.015)
    plt.axhline(y=0,color='gray',linewidth=0.5)
    #plt.grid()

    # Plot ax insert
    axins1 = ax1.inset_axes([0.39, 0.56, 0.35, 0.35])
    div = 18
    x1, x2, y1, y2 = 0.9, 1.00, -0.0025, 0.0025
    axins1.set_xlim(x1,x2)
    axins1.set_ylim(y1,y2)

    axins1.plot(mcc_start_z[:div+1],mcc_mean_z[:div+1],label='$\Delta C\leq$0')
    axins1.plot(mcc_start_z[div:],mcc_mean_z[div:],label='$\Delta C>$0')
    axins1.fill_between(mcc_start_z[:div+1],
                        mcc_mean_z[:div+1] - mcc_err_z[:div+1],
                        mcc_mean_z[:div+1] + mcc_err_z[:div+1],
                        color='C0',alpha=0.2,label='$\Delta C\leq0$, 95% CI')
    axins1.fill_between(mcc_start_z[div:],
                        mcc_mean_z[div:] - mcc_err_z[div:],
                        mcc_mean_z[div:] + mcc_err_z[div:],
                        color='C1',alpha=0.2,label='$\Delta C>0$, 95% CI')

    axins1.set_xticks((0.9,0.945,1.0))
    axins1.set_xticklabels((0.9,0.945,1.0))
    axins1.tick_params(labelsize=9)
    axins1.axhline(y=0,color='gray',linewidth=0.5)
    axins1.axvline(x=0.945,color='gray',linewidth=0.5)
    ax1.indicate_inset_zoom(axins1)

    handles, labels = axins1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.025),
               borderaxespad=0.,ncol=2)

    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    plt.show()
