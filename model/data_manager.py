# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:07:45 2020

@author: John Meluso
"""

import Organization as og
import numpy as np
import pickle
import os,sys
import datetime as dt


def save_mcc(data,file_name):
    """Saves data from test for the MCC simulation trials."""

    mcc = np.mean(data.socialization[:,:,0] \
                  + data.socialization[:,:,1],axis=1)
    inc = np.mean(data.socialization[:,:,2],axis=1)
    prf = data.performance_org
    dem = data.demographics

    with open(file_name,'wb') as file:
        np.save(file,mcc)
        np.save(file,inc)
        np.save(file,prf)
        np.save(file,dem)


def load_mcc(file_name):
    """Loads data from a saved MCC numpy trial file."""

    with open(file_name, 'rb') as file:
        mcc = np.load(file)
        inc = np.load(file)
        prf = np.load(file)
        dem = np.load(file)

    return mcc, inc, prf, dem


def generate_levels():
    """Generates levels for a (4,5) clique tree."""
    levels = np.zeros((781,1))
    levels[0] = 5
    levels[1:6] = 4
    levels[6:31] = 3
    levels[31:156] = 2
    levels[156:781] = 1
    return levels


def mean_level(demos,levels,pop):
    """Combines a demographic matrix describing what population each member is
    from at each point in time with a set of levels corresponding to the rank
    of each node position in the organization, given a population integer.
    Returns the mean rank for members of each population in the org."""
    is_pop = demos.__eq__(pop)
    pop_levels = (is_pop.T * levels).T
    pop_size = np.sum(is_pop,axis=1)
    return np.divide(np.sum(pop_levels,axis=1),pop_size)


def frac_level(demos,levels,pop):
    """Combines a demographic matrix describing what population each member is
    from at each point in time with a set of levels corresponding to the rank
    of each node position in the organization, given a population integer.
    For each level in the organization, returns the fraction of members who
    are from each population across all runs."""
    is_pop = demos.__eq__(pop)
    pop_levels = (is_pop.T * levels).T
    unique = np.unique(levels)
    size = demos.shape
    level_frac = np.zeros((size[0],len(unique)))
    for uu in np.arange(len(unique)):
        level_frac[:,uu] = np.count_nonzero(pop_levels == unique[uu], axis=1) \
            / np.count_nonzero(levels == unique[uu])
    return level_frac



def combine_mcc(directory,n_cases,n_runs,test=True):
    """Imports data from a specified directory string and combines them for
    analysis and plotting."""

    # Start timer
    t_start = dt.datetime.now()

    # Preset parameters for MCC simulation
    n_steps = 100
    n_pops = 2
    n_levels = 5
    levels = generate_levels()

    # Create structures for combining data
    mcc_all = np.zeros((n_cases,n_steps))
    inc_all = np.zeros((n_cases,n_steps))
    prf_all = np.zeros((n_cases,n_steps))
    dem_all = np.zeros((n_cases,n_steps,n_pops))
    lvl_all = np.zeros((n_cases,n_pops,n_steps,n_levels))

    # Iteratively open each file if it exists and import contents
    for ii in np.arange(n_cases):
        for jj in np.arange(n_runs):
            file = directory + f'case{ii:04}_run{jj:04}.npy'
            if os.path.exists(file):
                mcc, inc, prf, dem = load_mcc(file)
                mcc_all[ii,:] += mcc
                inc_all[ii,:] += inc
                prf_all[ii,:] += prf
                dem_all[ii,:,0] += mean_level(dem,levels,0)
                lvl_all[ii,0,:,:] += frac_level(dem,levels,0)
                if ii > 9:
                    dem_all[ii,:,1] += mean_level(dem,levels,1)
                    lvl_all[ii,1,:,:] += frac_level(dem,levels,1)
                else:
                    dem_all[ii,:,1] = dem_all[ii,:,0]
                    lvl_all[ii,1,:,:] = lvl_all[ii,0,:,:]

    # Average results
    if test: n_runs = 4
    mcc_mean = mcc_all/n_runs
    inc_mean = inc_all/n_runs
    prf_mean = prf_all/n_runs
    dem_mean = dem_all/n_runs
    lvl_mean = lvl_all/n_runs

    # Save results
    if test:
        loc = directory + 'test_results.npy'
    else:
        loc = directory + 'results.npy'
    with open(loc,'wb') as file:
        np.save(file,mcc_mean)
        np.save(file,inc_mean)
        np.save(file,prf_mean)
        np.save(file,dem_mean)
        np.save(file,lvl_mean)

    # Stop timer & print time
    t_stop = dt.datetime.now()
    print(t_stop - t_start)


def load_results(file_name):
    """Loads results from a saved MCC numpy results file."""

    with open(file_name, 'rb') as file:
        mcc = np.load(file)
        inc = np.load(file)
        prf = np.load(file)
        dem = np.load(file)
        lvl = np.load(file)

    return mcc, inc, prf, dem, lvl


if __name__ == '__main__':

    name = 'culture_sim_exec001'

    if sys.platform.startswith('linux'):
        loc = '/gpfs1/home/j/m/jmeluso/culture_sim/data/' + name + '/'
        combine_mcc(loc,640,100,False)
    else:
        loc = '../data/' + name + '/'
        combine_mcc(loc,640,100)
        mcc, inc, prf, dem, lvl = load_results(loc + 'test_results.npy')




