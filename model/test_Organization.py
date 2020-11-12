# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:21:10 2020

@author: John Meluso
"""

import sys
import numpy as np
import Organization as og
import data_manager as dm
from numpy.random import default_rng
import matplotlib.pyplot as plt
import datetime as dt

# Create the random number generator
rng = default_rng()

def add_test_pops(case=3):
    """Adds test populations"""

    pops = []  # Create empty population array

    # CASE 1: One uniform population with a 1-D variable spectrum
    if case == 1:
        pops.append(og.Population(aff_dist="uniform_2var"))

    # CASE 2: Two beta-distributed populations with a 1-D variable spectrum
    elif case == 2:

        # Add generic org of beta culture distribution
        pops.append(og.Population(starting=0.9,
                                  hires=0.7,
                                  aff_dist="beta_2var",
                                  aff_sim=0.45,
                                  aff_perf=0.45,
                                  aff_inc=0.1))

        # Add generic org of beta culture distribution
        pops.append(og.Population(starting=0.1,
                                  hires=0.3,
                                  aff_dist="beta_2var",
                                  aff_sim=0.25,
                                  aff_perf=0.25,
                                  aff_inc=0.5))

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
                                  aff_dist="uniform_3var"))

    # CASE 5: One beta-distributed population with a 1-D variable spectrum
    elif case == 5:

        # Add generic org of beta culture distribution
        pops.append(og.Population(aff_dist="beta_2var",
                                  aff_sim=0.3,
                                  aff_perf=0.3,
                                  aff_inc=0.4))

    return pops


def mean_level(demos,levels,pop):
    """Combines a demographic matrix describing what population each member is
    from at each point in time with a set of levels corresponding to the rank
    of each node position in the organization, given a population integer."""
    is_pop = demos.__eq__(pop)
    pop_levels = (is_pop.T * levels).T
    pop_size = np.sum(is_pop,axis=1)
    return np.divide(np.sum(pop_levels,axis=1),pop_size)


if __name__ == '__main__':

    # Specify number of steps to run simulation
    n_steps = 100
    case = 2
    plots = True

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

    # Import levels
    level = dm.generate_levels()

    # Save test results
    if sys.platform.startswith('linux'):
        save_dir = '/users/j/m/jmeluso/scratch/culture_sim/data/'
    else:
        save_dir = '../data/'
    dm.save_mcc(history, save_dir + 'test_results.npy')

    # Plot test results by case
    if plots:
        if case == 1:

            # Grab values for culture plot
            x_values = np.arange(n_steps)
            mcc = np.mean(history.socialization[:,:,0] \
                          + history.socialization[:,:,1],axis=1)
            inc = np.mean(history.socialization[:,:,2],axis=1)

            # Plot culture results
            plt.plot(x_values,mcc,label='Contest')
            plt.plot(x_values,inc,label='Inclusiveness')
            plt.ylim(0, 1)
            plt.legend()
            plt.show()

        elif case == 2:

            # Grab values for culture plot
            x_values = np.arange(n_steps)
            mcc = np.mean(history.socialization[:,:,0] \
                          + history.socialization[:,:,1],axis=1)
            inc = np.mean(history.socialization[:,:,2],axis=1)
            prf = history.performance_org
            pop1 = mean_level(history.demographics,level,0)
            pop2 = mean_level(history.demographics,level,1)

            # Create figure
            plt.figure(figsize=(9,3))

            # Plot culture results
            plt.subplot(131)
            plt.plot(x_values,mcc,label='Contest')
            plt.plot(x_values,inc,label='Inclusiveness')
            plt.ylim(0, 1)
            plt.legend()

            # Plot performance results
            plt.subplot(132)
            plt.plot(x_values,prf,label='Performance')
            plt.ylim(0, 1)
            plt.legend()

            # Plot mean level by population rank
            plt.subplot(133)
            plt.plot(x_values,pop1,label='Pop1')
            plt.plot(x_values,pop2,label='Pop2')
            plt.legend()

            # Show figure
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
            plt.ylim(0, 1)
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
            plt.ylim(0, 1)
            plt.legend()
            plt.show()

        elif case == 5:

            # Grab values for culture plot
            x_values = np.arange(n_steps)
            mcc = np.mean(history.socialization[:,:,0] \
                          + history.socialization[:,:,1],axis=1)
            inc = np.mean(history.socialization[:,:,2],axis=1)
            prf = 2*history.performance_branch[:,0] \
                - history.performance_indiv[:,0]

            # Create figure
            plt.figure(figsize=(6,3))

            # Plot culture results
            plt.subplot(121)
            plt.plot(x_values,mcc,label='Contest')
            plt.plot(x_values,inc,label='Inclusiveness')
            plt.ylim(0, 1)
            plt.legend()

            # Plot performance results
            plt.subplot(122)
            plt.plot(x_values,prf,label='Performance')
            plt.ylim(0, 1)
            plt.legend()

            # Show figure
            plt.show()

        else:
            print("Not a valid case.")
