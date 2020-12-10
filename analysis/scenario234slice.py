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

# Import results
mcc, inc, prf, dem, lvl \
        = dm.load_exec001_results('../data/culture_sim_exec001/results.npy')

# Load MCC Cases & Scenarios
cases, scen, paired, indeces = da.load_exec001_conditions()

# Combine scenarios
combined = indeces[2]
for ii in indeces[3]: combined.append(ii)
for ii in indeces[4]: combined.append(ii)

# Get case-specific data
slc = []

# Get a slice of the population culture space
for ii in combined:
    try:
        if (cases[ii][index_p1_culture] == 0.8) \
            and (cases[ii][index_p1_start] == 0.8) \
            and (cases[ii][index_p1_hire] == 0.8):
            slc.append(ii)
    except IndexError:
        if (cases[ii][index_p1_culture] == 0.8):
            slc.append(ii)


p2c_start = []
diff_culture = []
diff_prf = []

for ii in slc:
    try:
        p2c_start.append(1 - cases[ii][index_p2_culture])
    except IndexError:
        p2c_start.append(1 - cases[ii][index_p1_culture])
    diff_culture.append(mcc[ii,-1] - mcc[ii,0])
    diff_prf.append(prf[ii,-1] - prf[ii,0])

# Convert to Numpy Arrays
p2c_start = np.array(p2c_start)
diff_culture = np.array(diff_culture)
diff_prf = np.array(diff_prf)

# Get sort order indeces, & sort by them
p2c_index = np.argsort(p2c_start)
p2c_start = np.take_along_axis(p2c_start,p2c_index,axis=0)
diff_culture = np.take_along_axis(diff_culture,p2c_index,axis=0)
diff_prf = np.take_along_axis(diff_prf,p2c_index,axis=0)

# Define constants
n_steps = 100
x_values = np.arange(n_steps)

if plot_num == 1:

    # Create the plot
    plt.figure(figsize=(7,4))
    plt.suptitle('Scenario 234 Slice')

    # Plot culture results
    plt.subplot(1,2,1)
    for ii in slc:
        try:
            plt.plot(x_values,mcc[ii,:],
                     label=f'MCC={cases[ii][index_p2_culture]}')
        except IndexError:
            plt.plot(x_values,mcc[ii,:],
                     label=f'MCC={cases[ii][index_p1_culture]}')
    plt.xlabel('Turns')
    plt.ylabel('Avg. Attribute Prevalence')
    plt.ylim(0, 1)

    # Plot performance results
    plt.subplot(1,2,2)
    for ii in slc:
        try:
            plt.plot(x_values,prf[ii,:],
                     label=f'MCC={cases[ii][index_p2_culture]}')
        except IndexError:
            plt.plot(x_values,prf[ii,:],
                     label=f'MCC={cases[ii][index_p1_culture]}')
    plt.xlabel('Turns')
    plt.ylabel('Organization Performance')
    plt.ylim(0, 1)
    plt.legend(loc='upper left',bbox_to_anchor=(1.05, 1),borderaxespad=0.)

    # Show figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

elif plot_num == 2:

    # Plot culture results
    plt.figure
    plt.title('Scenario 234 Slice')
    plt.plot(p2c_start, diff_culture, label='Culture Change')
    plt.plot(p2c_start, diff_prf, label='Performance Change')
    plt.xlabel('Mean Inclusiveness of Population 2')
    plt.ylabel('Parameter Change at End')
    plt.legend()




