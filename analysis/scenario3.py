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
scen_num = 3

# Import results
mcc, inc, prf, dem, lvl \
        = dm.load_exec001_results('../data/culture_sim_exec001/results.npy')

# Load MCC Cases & Scenarios
cases, scen, paired, indeces = da.load_exec001_conditions()

# Get case-specific data
scen_cases = [cases[ii] for ii in indeces[scen_num]]
scen_mcc = [mcc[ii] for ii in indeces[scen_num]]
scen_inc = [inc[ii] for ii in indeces[scen_num]]
scen_prf = [prf[ii] for ii in indeces[scen_num]]
scen_dem = [dem[ii] for ii in indeces[scen_num]]
scen_lvl = [lvl[ii] for ii in indeces[scen_num]]

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
for jj in np.arange(len(scen_cases)):

    # Get indeces
    x_index = np.argwhere(x==scen_cases[jj][index_p1_culture])
    y_index = np.argwhere(y==scen_cases[jj][index_p2_culture])
    s_index = np.argwhere(s==scen_cases[jj][index_p1_start])

    # Place data in numpy array
    z_mcc_abs[x_index,y_index,s_index] = scen_mcc[jj][-1]
    z_mcc_del[x_index,y_index,s_index] = scen_mcc[jj][-1] - scen_mcc[jj][0]

if plot_num == 1:

    x_values = np.arange(0.45,1,0.1)
    y_values = np.arange(0.05,1,0.1)

    fig, axs = plt.subplots(2,5)
    plt.subplot(2,5,1)
    plt.pcolormesh(x_values,y_values,z_mcc_abs[:,:,0].T,vmin=0,vmax=1)

    plt.subplot(2,5,2)
    plt.pcolormesh(x_values,y_values,z_mcc_abs[:,:,1].T,vmin=0,vmax=1)

    plt.subplot(2,5,3)
    plt.pcolormesh(x_values,y_values,z_mcc_abs[:,:,2].T,vmin=0,vmax=1)

    plt.subplot(2,5,4)
    plt.pcolormesh(x_values,y_values,z_mcc_abs[:,:,3].T,vmin=0,vmax=1)

    plt.subplot(2,5,5)
    plt.pcolormesh(x_values,y_values,z_mcc_abs[:,:,4].T,vmin=0,vmax=1)
    plt.colorbar()

    plt.subplot(2,5,6)
    plt.pcolormesh(x_values,y_values,z_mcc_del[:,:,0].T,vmin=-0.2,vmax=0)

    plt.subplot(2,5,7)
    plt.pcolormesh(x_values,y_values,z_mcc_del[:,:,1].T,vmin=-0.2,vmax=0)

    plt.subplot(2,5,8)
    plt.pcolormesh(x_values,y_values,z_mcc_del[:,:,2].T,vmin=-0.2,vmax=0)

    plt.subplot(2,5,9)
    plt.pcolormesh(x_values,y_values,z_mcc_del[:,:,3].T,vmin=-0.2,vmax=0)

    plt.subplot(2,5,10)
    plt.pcolormesh(x_values,y_values,z_mcc_del[:,:,4].T,vmin=-0.2,vmax=0)

    plt.colorbar()
    plt.show()







