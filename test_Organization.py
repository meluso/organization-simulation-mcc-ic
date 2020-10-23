# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:21:10 2020

@author: John Meluso
"""

import numpy as np
import Organization as og
from numpy.random import default_rng
import matplotlib.pyplot as plt

# Create the random number generator
rng = default_rng()

def add_test_pops():
    """Adds test populations"""

    pops = []  # Create empty population array

    # # CASE 1: Only 1 Uniform Population
    # pops.append(og.Population(starting=1.0,
    #                           hires=1.0,
    #                           aff_dist="uniform"))

    # CASE 2: Two beta-distributed populations
    pops.append(og.Population(starting=0.9,
                              hires=0.5,
                              aff_dist="beta",
                              aff_sim=0.45,
                              aff_perf=0.45,
                              aff_inc=0.1))

    # Add generic org of beta culture distribution
    pops.append(og.Population(starting=0.1,
                              hires=0.5,
                              aff_dist="beta",
                              aff_sim=0.2,
                              aff_perf=0.2,
                              aff_inc=0.6))



    return pops


if __name__ == '__main__':

    # Specify number of steps to run simulation
    n_steps = 100

    # Create organization and empty population array
    org_test = og.Organization()

    # Add the test populations
    pops = add_test_pops()

    # Populate the organization with the culture, performance, and populations
    org_test.fill_org(pops)

    # Run organization for n iterations
    org_test.org_step(n_steps)

    # Return results
    history = org_test.return_results()

    # Grab values for culture plot
    x_values = np.arange(n_steps)
    mcc = np.mean(history.socialization[:,:,0] + history.socialization[:,:,1],axis=1)
    inc = np.mean(history.socialization[:,:,2],axis=1)

    # Plot culture results
    plt.plot(x_values,mcc,label='Contest')
    plt.plot(x_values,inc,label='Inclusiveness')
    plt.legend()
    plt.show()
