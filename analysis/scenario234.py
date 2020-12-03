# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:59:35 2020

@author: John Meluso
"""

import numpy as np
import matplotlib.pyplot as plt
import data_manager as dm
import analysis.data_analysis as da


#Specify plot to generate
plot_num = 1
scen_num = [2,3,4]

# Import results
mcc, inc, prf, dem, lvl \
        = dm.load_exec001_results('./data/culture_sim_exec001/results.npy')

# Load MCC Cases & Scenarios
cases, scen, paired, indeces = da.load_exec001_conditions()

# Combine scenarios
combined = []
for ii in scen_num:
    for jj in indeces[ii]:
        combined.append(jj)

# Get case-specific data
comb_cases = [cases[ii] for ii in combined]
comb_mcc = [mcc[ii] for ii in combined]
comb_inc = [inc[ii] for ii in combined]
comb_prf = [prf[ii] for ii in combined]
comb_dem = [dem[ii] for ii in combined]
comb_lvl = [lvl[ii] for ii in combined]

# Create simulation spec parameters
index_pops = 0
index_mode = 1
index_p1_culture = 2
index_p2_culture = 3
index_p1_start = 4
index_p1_hire = 5

# Create results parameters
index_mcc = 0
index_inc = 1
index_prf = 2
index_d1  = 3
index_d2  = 4

# Define constants
n_steps = 100
x = np.round(np.linspace(0.5,1.0,5,endpoint=False),1)
y = np.round(np.linspace(0.1,1.0,9,endpoint=False),1)
s = np.round(np.linspace(0.5,1.0,5,endpoint=False),1)
z_mcc_abs = np.zeros((len(x),len(y),5))
z_mcc_del = np.zeros((len(x),len(y),5))
z_prf_abs = np.zeros((len(x),len(y),5))
z_prf_del = np.zeros((len(x),len(y),5))

for jj in np.arange(len(comb_cases)):

    try:

        # Get indeces
        x_index = np.argwhere(x==comb_cases[jj][index_p1_culture])
        y_index = np.argwhere(y==comb_cases[jj][index_p2_culture])
        s_index = np.argwhere(s==comb_cases[jj][index_p1_start])

        # Place data in numpy array
        z_mcc_abs[x_index,y_index,s_index] = comb_mcc[jj][-1]
        z_mcc_del[x_index,y_index,s_index] = comb_mcc[jj][-1] - comb_mcc[jj][0]
        z_prf_abs[x_index,y_index,s_index] = comb_prf[jj][-1]
        z_prf_del[x_index,y_index,s_index] = comb_prf[jj][-1] - comb_prf[jj][0]

    except IndexError:
        for ss in np.arange(5):

            # Get indeces
            x_index = np.argwhere(x==comb_cases[jj][index_p1_culture])
            y_index = np.argwhere(y==comb_cases[jj][index_p1_culture])
            s_index = ss

            # Place data in numpy array
            z_mcc_abs[x_index,y_index,s_index] = comb_mcc[jj][-1]
            z_mcc_del[x_index,y_index,s_index] \
                = comb_mcc[jj][-1] - comb_mcc[jj][0]
            z_prf_abs[x_index,y_index,s_index] = comb_prf[jj][-1]
            z_prf_del[x_index,y_index,s_index] \
                = comb_prf[jj][-1] - comb_prf[jj][0]


if plot_num == 1:

    x_values = np.arange(0.45,1,0.1)
    y_values = np.arange(0.05,1,0.1)

    n_rows = 2
    n_cols = 5

    cmap_abs = plt.get_cmap('binary')
    cmap_del = plt.get_cmap('binary')

    fig, axs = plt.subplots(n_rows,n_cols,figsize=(15,4))

    for row in np.arange(n_rows):

        for col in np.arange(n_cols):
            ax = axs[row,col]
            if row == 0:
                pcm = ax.pcolormesh(x_values,y_values,z_mcc_abs[:,:,col].T,
                                    vmin=0,vmax=1,cmap=cmap_abs)
                ax.set(title = f'Pop1 Frac = {(col+5)/10}')
                ax.get_xaxis().set_visible(False)
            else:
                pcm = ax.pcolormesh(x_values,y_values,z_mcc_del[:,:,col].T,
                                    vmin=-0.2,vmax=0,cmap=cmap_del)
            if not(col == 0):
                ax.get_yaxis().set_visible(False)
            else:
                if row == 0:
                    ax.set(ylabel='End Culture')
                else:
                    ax.set(ylabel='Culture Change')


        fig.colorbar(pcm, ax=axs[row,:])

    plt.suptitle('Organization Final Culture with Two Populations')
    plt.show()

if plot_num == 2:

    x_values = np.arange(0.5,1,0.1)
    y_values = np.arange(0.1,1,0.1)

    n_rows = 2
    n_cols = 5

    cmap_abs = plt.get_cmap('binary')
    cmap_del = plt.get_cmap('binary')

    fig, axs = plt.subplots(n_rows,n_cols,figsize=(12,4))

    for row in np.arange(n_rows):

        for col in np.arange(n_cols):
            ax = axs[row,col]
            if row == 0:
                pcm = ax.contour(x_values,y_values,z_mcc_abs[:,:,col].T,
                                    vmin=0,vmax=1,cmap=cmap_abs)
                ax.clabel(pcm)
                ax.set(title = f'Pop1 Frac = {(col+5)/10}')
                ax.get_xaxis().set_visible(False)
            else:
                pcm = ax.contour(x_values,y_values,z_mcc_del[:,:,col].T,
                                    vmin=-0.2,vmax=0,cmap=cmap_del)
                ax.clabel(pcm)
            if not(col == 0):
                ax.get_yaxis().set_visible(False)
            else:
                if row == 0:
                    ax.set(ylabel='End Culture')
                else:
                    ax.set(ylabel='Culture Change')

    plt.suptitle('Organization Final Culture with Two Populations')
    plt.show()

elif plot_num == 3:

    x_values = np.arange(0.45,1,0.1)
    y_values = np.arange(0.05,1,0.1)

    n_rows = 2
    n_cols = 5

    cmap_abs = plt.get_cmap('binary')
    cmap_del = plt.get_cmap('binary')

    fig, axs = plt.subplots(n_rows,n_cols,figsize=(15,4))

    for row in np.arange(n_rows):

        for col in np.arange(n_cols):
            ax = axs[row,col]
            if row == 0:
                pcm = ax.pcolormesh(x_values,y_values,z_prf_abs[:,:,col].T,
                                    vmin=0,vmax=1,cmap=cmap_abs)
                ax.set(title = f'Pop1 Frac = {(col+5)/10}')
                ax.get_xaxis().set_visible(False)
            else:
                pcm = ax.pcolormesh(x_values,y_values,z_prf_del[:,:,col].T,
                                    vmin=-0.2,vmax=0.2,cmap=cmap_del)
            if not(col == 0):
                ax.get_yaxis().set_visible(False)
            else:
                if row == 0:
                    ax.set(ylabel='End Performance')
                else:
                    ax.set(ylabel='Performance Change')


        fig.colorbar(pcm, ax=axs[row,:])

    plt.suptitle('Organization Final Performance with Two Populations')
    plt.show()

elif plot_num == 4:

    x_values = np.arange(0.5,1,0.1)
    y_values = np.arange(0.1,1,0.1)

    n_rows = 2
    n_cols = 5

    cmap_abs = plt.get_cmap('binary')
    cmap_del = plt.get_cmap('binary')

    fig, axs = plt.subplots(n_rows,n_cols,figsize=(12,4))

    for row in np.arange(n_rows):

        for col in np.arange(n_cols):
            ax = axs[row,col]
            if row == 0:
                pcm = ax.contour(x_values,y_values,z_prf_abs[:,:,col].T,
                                    vmin=0,vmax=1,cmap=cmap_abs)
                ax.clabel(pcm)
                ax.set(title = f'Pop1 Frac = {(col+5)/10}')
                ax.get_xaxis().set_visible(False)
            else:
                pcm = ax.contour(x_values,y_values,z_prf_del[:,:,col].T,
                                    vmin=-0.2,vmax=0.2,cmap=cmap_del)
                ax.clabel(pcm)
            if not(col == 0):
                ax.get_yaxis().set_visible(False)
            else:
                if row == 0:
                    ax.set(ylabel='End Performance')
                else:
                    ax.set(ylabel='Performance Change')

    plt.suptitle('Organization Final Performance with Two Populations')
    plt.show()





