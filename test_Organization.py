# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:21:10 2020

@author: John Meluso
"""

import numpy as np
import Organization as og
from numpy.random import default_rng
import matplotlib.pyplot as plt
import datetime as dt
import pickle

# Create the random number generator
rng = default_rng()

def add_test_pops(case=3):
    """Adds test populations"""

    pops = []  # Create empty population array

    # CASE 1: One uniform population with a 1-D variable spectrum
    if case == 1:
        pops.append(og.Population(starting=1.0,
                                  hires=1.0,
                                  aff_dist="linear_2var"))

    # CASE 2: Two beta-distributed populations with a 1-D variable spectrum
    elif case == 2:

        # Add generic org of beta culture distribution
        pops.append(og.Population(starting=0.9,
                                  hires=0.7,
                                  aff_dist="beta_2var",
                                  aff_sim=0.35,
                                  aff_perf=0.35,
                                  aff_inc=0.3))

        # Add generic org of beta culture distribution
        pops.append(og.Population(starting=0.1,
                                  hires=0.3,
                                  aff_dist="beta_2var",
                                  aff_sim=0.2,
                                  aff_perf=0.2,
                                  aff_inc=0.6))

    # CASE 3: Two beta-distributed populations with a 2-D variable triangle
    elif case == 3:

        # Add generic org of beta culture distribution
        pops.append(og.Population(starting=0.9,
                                  hires=0.5,
                                  aff_dist="beta_3var",
                                  aff_sim=0.15,
                                  aff_perf=0.7,
                                  aff_inc=0.15))

        # Add generic org of beta culture distribution
        pops.append(og.Population(starting=0.1,
                                  hires=0.5,
                                  aff_dist="beta_3var",
                                  aff_sim=0.3,
                                  aff_perf=0.2,
                                  aff_inc=0.5))

    # CASE 4: One uniform population with a 2-D variable triangle
    elif case == 4:
        pops.append(og.Population(starting=1.0,
                                  hires=1.0,
                                  aff_dist="linear_3var"))

    return pops


if __name__ == '__main__':

    # Specify number of steps to run simulation
    n_steps = 100
    case = 2

    # Start timer
    t_start = dt.datetime.now()

    # Create organization and empty population array
    org_test = og.Organization()

    # Add the test populations
    pops = add_test_pops(case)

    # Populate the organization with the culture, performance, and populations
    org_test.fill_org(pops)

    # Run organization for n iterations
    org_test.org_step(n_steps)

    # Return results
    history = org_test.return_results()

    # Stop timer & print time
    t_stop = dt.datetime.now()
    print(t_stop - t_start)

    # Plot test results by case
    if case == 1:

        # Grab values for culture plot
        x_values = np.arange(n_steps)
        mcc = np.mean(history.socialization[:,:,0] + history.socialization[:,:,1],axis=1)
        inc = np.mean(history.socialization[:,:,2],axis=1)

        # Plot culture results
        plt.plot(x_values,mcc,label='Contest')
        plt.plot(x_values,inc,label='Inclusiveness')
        plt.legend()
        plt.show()

    elif case == 2:

        # Grab values for culture plot
        x_values = np.arange(n_steps)
        mcc = np.mean(history.socialization[:,:,0] + history.socialization[:,:,1],axis=1)
        inc = np.mean(history.socialization[:,:,2],axis=1)

        # Plot culture results
        plt.plot(x_values,mcc,label='Contest')
        plt.plot(x_values,inc,label='Inclusiveness')
        plt.legend()
        plt.show()

    elif case == 3:

        # Grab values for culture plot
        x_values = np.arange(n_steps)
        sim = np.mean(history.socialization[:,:,0],axis=1)
        perf = np.mean(history.socialization[:,:,1],axis=1)
        inc = np.mean(history.socialization[:,:,2],axis=1)

        # Plot culture results
        plt.plot(x_values,sim,label='Similarity')
        plt.plot(x_values,perf,label='Performance')
        plt.plot(x_values,inc,label='Inclusiveness')
        plt.legend()
        plt.show()

    elif case == 4:

        # Grab values for culture plot
        x_values = np.arange(n_steps)
        sim = np.mean(history.socialization[:,:,0],axis=1)
        perf = np.mean(history.socialization[:,:,1],axis=1)
        inc = np.mean(history.socialization[:,:,2],axis=1)

        # Plot culture results
        plt.plot(x_values,sim,label='Similarity')
        plt.plot(x_values,perf,label='Performance')
        plt.plot(x_values,inc,label='Inclusiveness')
        plt.legend()
        plt.show()

    else:
        print("Not a valid case.")
