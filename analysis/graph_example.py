# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 18:40:07 2020

@author: John Meluso
"""

import numpy as np
import matplotlib.pyplot as plt
import data_manager as dm
import analysis.data_analysis as da

# Select scenario
scenario=1

# Create figure
plt.figure(figsize=(7.5,5),dpi=300)

# Import results
if scenario == 1:
    culture, org_prf, org_dem \
        = dm.load_exec002_data('../data/example_scenario1.npy')
    #plt.suptitle('Example Run w/ Uniform Culture Distribution')
else:
    culture, org_prf, org_dem \
        = dm.load_exec002_data('../data/example_scenario2.npy')
    #plt.suptitle('Example Run w/ Beta Culture Distribution & Mean = 0.8')


# Import levels
levels = dm.generate_levels()

# Grab values for culture plot
n_steps = 100
# Grab values for culture plot
x_values = np.arange(n_steps)
mcc = np.mean(culture[:,:,0] + culture[:,:,1],axis=1)
inc = np.mean(culture[:,:,2],axis=1)
prf = org_prf

# Plot culture results
plt.subplot(131)
plt.plot(x_values,100*mcc,label='Contest-Orientation')
plt.plot(x_values,100*inc,label='Inclusiveness')
plt.xlabel('Turns')
plt.ylabel('Attribute Prevalence (%)')
plt.ylim(0, 100)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.2),borderaxespad=0.)

# Plot inclusiveness at each level
plt.subplot(132)
culture_by_level = dm.mean_culture_level(levels,
                                         culture[:,:,0] + culture[:,:,1])
plt.plot(x_values,100*culture_by_level[:,0],label='Level 1')
plt.plot(x_values,100*culture_by_level[:,1],label='Level 2')
plt.plot(x_values,100*culture_by_level[:,2],label='Level 3')
plt.plot(x_values,100*culture_by_level[:,3],label='Level 4')
plt.plot(x_values,100*culture_by_level[:,4],label='Level 5')
plt.xlabel('Turns')
plt.ylabel('Contest-Orientation Prevalence (%)')
plt.ylim(0, 100)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.2),borderaxespad=0.)

# Plot performance results
plt.subplot(133)
plt.plot(x_values,prf,label='Performance')
plt.xlabel('Turns')
plt.ylabel('Organization Performance')
plt.ylim(0, 1)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.2),borderaxespad=0.)

# Show figure
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()