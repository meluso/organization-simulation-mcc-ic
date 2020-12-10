# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:07:45 2020

@author: John Meluso
"""

import os,sys
import numpy as np
import datetime as dt


#%% Shared Methods

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


def mean_culture_level(levels,culture):
    """Returns the average culture for each level at each point in time."""
    unique_levels, per_level = np.unique(levels, return_counts=True)
    is_level = levels.__eq__(unique_levels)
    return np.divide(np.matmul(culture,is_level),per_level)


#%% Exec001 Methods

def save_exec001_data(data,file_name):
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


def load_exec001_data(file_name):
    """Loads data from a saved MCC numpy trial file."""

    with open(file_name, 'rb') as file:
        mcc = np.load(file)
        inc = np.load(file)
        prf = np.load(file)
        dem = np.load(file)

    return mcc, inc, prf, dem


def cases_exec001():
    """Create one instance of each combination of the MCC simulation run
    parameters. Case Structure appears as follows:
        [n_pops,pop_mode,pop1_culture,pop2_culture,pop_start,pop_hire]
    """

    cases = []

    """Create 1 population uniform cases"""
    n_pops = 1
    pop_mode = "uniform_2var"
    new_case = [n_pops,pop_mode]
    cases.append(new_case)

    """Create 1 population beta cases"""
    n_pops = 1
    pop_mode = "beta_2var"
    pop1_culture = np.round(np.linspace(0.1,1.0,9,endpoint=False),1)
    for cc in pop1_culture:
        new_case = [n_pops,pop_mode,cc]
        cases.append(new_case)

    """Create 2 population beta cases"""
    n_pops = 2
    pop_mode = "beta_2var"
    pop_hire = np.round(np.linspace(0.5,1.0,5,endpoint=False),1)
    pop1_start_limit = 1.0  # for arange in loop
    pop1_culture = np.round(np.linspace(0.5,1.0,5,endpoint=False),1)
    pop2_culture = np.round(np.linspace(0.1,1.0,9,endpoint=False),1)

    # Loop through all starting fractions and hiring fractions
    for hh in pop_hire:
        pop1_steps = int(np.round((pop1_start_limit - hh)/0.1))
        for ss in np.round(np.linspace(hh,pop1_start_limit,pop1_steps,
                              endpoint=False),1):

            # Loop through full rectangle of population 1 and 2 cultures
            for cc in pop1_culture:
                for dd in pop2_culture:
                    if not(abs(cc - dd) < 1E-4):

                        # Construct cases
                        new_case = [n_pops,pop_mode,cc,dd,ss,hh]
                        cases.append(new_case)

            # Loop through special cases for populations 1 and 2 cultures
            # w/ 0.3,0.5 and 0.3,0.7 for pop1_culture,pop2_culture resp.
            new_case = [n_pops,pop_mode,0.3,0.5,ss,hh]
            cases.append(new_case)
            new_case = [n_pops,pop_mode,0.3,0.7,ss,hh]
            cases.append(new_case)

    return cases


def combine_exec001(directory,n_cases,n_runs,test=True):
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
    lvl_all = np.zeros((n_cases,n_steps,n_pops,n_levels))

    # Iteratively open each file if it exists and import contents
    for ii in np.arange(n_cases):
        for jj in np.arange(n_runs):
            file = directory + f'case{ii:04}_run{jj:04}.npy'
            if os.path.exists(file):
                mcc, inc, prf, dem = load_exec001_data(file)
                mcc_all[ii,:] += mcc
                inc_all[ii,:] += inc
                prf_all[ii,:] += prf
                dem_all[ii,:,0] += mean_level(dem,levels,0)
                lvl_all[ii,:,0,:] += frac_level(dem,levels,0)
                if ii > 9:
                    dem_all[ii,:,1] += mean_level(dem,levels,1)
                    lvl_all[ii,:,1,:] += frac_level(dem,levels,1)
                else:
                    dem_all[ii,:,1] = dem_all[ii,:,0]
                    lvl_all[ii,:,1,:] = lvl_all[ii,0,:,:]

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


def load_exec001_results(file_name):
    """Loads results from a saved MCC numpy results file."""

    with open(file_name, 'rb') as file:
        mcc = np.load(file)
        inc = np.load(file)
        prf = np.load(file)
        dem = np.load(file)
        lvl = np.load(file)

    return mcc, inc, prf, dem, lvl


#%% Exec002 Methods

def save_exec002_data(data,file_name):
    """Saves data from a simulation execution."""

    culture = data.socialization
    org_prf = data.performance_org
    org_dem = data.demographics

    with open(file_name,'wb') as file:
        np.save(file,culture)
        np.save(file,org_prf)
        np.save(file,org_dem)


def load_exec002_data(file_name):
    """Loads data from a simulation execution."""

    with open(file_name, 'rb') as file:
        culture = np.load(file)
        org_prf = np.load(file)
        org_dem = np.load(file)

    return culture, org_prf, org_dem


def cases_exec002():
    """Create one instance of each combination of the simulation run
    parameters. Case Structure appears as follows:
        [n_pops,pop_mode,pop1_culture,pop2_culture,pop_start,pop_hire]
    """

    cases = []

    """Create 1 population uniform cases"""
    n_pops = 1
    pop_mode = "uniform_2var"
    new_case = [n_pops,pop_mode]
    cases.append(new_case)

    """Create 1 population beta cases"""
    n_pops = 1
    pop_mode = "beta_2var"
    pop1_culture = np.round(np.linspace(0.01,1.0,99,endpoint=False),2)
    for cc in pop1_culture:
        new_case = [n_pops,pop_mode,cc]
        cases.append(new_case)

    return cases


def combine_exec002(directory,n_cases,n_runs,test=True):
    """Imports data from a specified directory string and combines them for
    analysis and plotting."""

    # Start timer
    t_start = dt.datetime.now()

    # Preset parameters for MCC simulation
    n_steps = 100
    n_levels = 5
    levels = generate_levels()

    # # Create csv for combining data
    # cols = ['case','run','step','lvl','org_mcc','org_prf','lvl_mcc']
    # n_cols = len(cols)
    # n_rows = n_cases*n_runs*n_steps*n_levels
    # csv_mcc = np.zeros((n_cols,n_rows))
    # csv_prf = np.zeros((n_cols,n_rows))
    # csv_lvl = np.zeros((n_cols,n_rows))

    # Create array for combining data
    arr_mcc = np.zeros((n_cases,n_runs,n_steps))
    arr_prf = np.zeros((n_cases,n_runs,n_steps))
    arr_lvl = np.zeros((n_cases,n_runs,n_steps,n_levels))

    # Iteratively open each file if it exists and import contents
    for ii in np.arange(n_cases):
        for jj in np.arange(n_runs):
            file = directory + f'case{ii:04}_run{jj:04}.npy'
            if os.path.exists(file):

                # Get data from file
                culture, org_prf, org_dem = load_exec002_data(file)

                # Write results to csv

                # Write results to array
                xy = culture[:,:,0] + culture[:,:,1]
                arr_mcc[ii,jj,:] = np.mean(xy,axis=1)
                arr_prf[ii,jj,:] = org_prf
                arr_lvl[ii,jj,:,:] = mean_culture_level(levels,xy)

    # Save results
    if test:
        loc = directory + 'test_results.npy'
    else:
        loc = directory + 'results.npy'
    with open(loc,'wb') as file:
        np.save(file,arr_mcc)
        np.save(file,arr_prf)
        np.save(file,arr_lvl)

    # Stop timer & print time
    t_stop = dt.datetime.now()
    print(t_stop - t_start)


def load_exec002_results(file_name):
    """Loads results from a saved MCC numpy results file."""

    with open(file_name, 'rb') as file:
        mcc = np.load(file)
        prf = np.load(file)
        lvl = np.load(file)

    return mcc, prf, lvl


#%% Exec003 Methods

def save_exec003_data(data,file_name):
    """Saves data from a simulation execution."""

    culture = data.socialization
    org_prf = data.performance_org
    org_dem = data.demographics

    with open(file_name,'wb') as file:
        np.save(file,culture)
        np.save(file,org_prf)
        np.save(file,org_dem)


def load_exec003_data(file_name):
    """Loads data from a simulation execution."""

    with open(file_name, 'rb') as file:
        culture = np.load(file)
        org_prf = np.load(file)
        org_dem = np.load(file)

    return culture, org_prf, org_dem


def cases_exec003():
    """Create one instance of each combination of the simulation run
    parameters. Case Structure appears as follows:
        [n_pops,pop_mode,pop1_culture,pop2_culture,pop_start,pop_hire]
    """

    cases = []

    """Create 1 population beta cases"""
    n_pops = 1
    pop_mode = "beta_2var"
    pop1_culture = np.round(np.linspace(0.85,1.0,30,endpoint=False),3)
    for cc in pop1_culture:
        new_case = [n_pops,pop_mode,cc]
        cases.append(new_case)

    return cases


def combine_exec003(directory,n_cases,n_runs,test=True):
    """Imports data from a specified directory string and combines them for
    analysis and plotting."""

    # Start timer
    t_start = dt.datetime.now()

    # Preset parameters for MCC simulation
    n_steps = 100
    n_levels = 5
    levels = generate_levels()

    # # Create csv for combining data
    # cols = ['case','run','step','lvl','org_mcc','org_prf','lvl_mcc']
    # n_cols = len(cols)
    # n_rows = n_cases*n_runs*n_steps*n_levels
    # csv_mcc = np.zeros((n_cols,n_rows))
    # csv_prf = np.zeros((n_cols,n_rows))
    # csv_lvl = np.zeros((n_cols,n_rows))

    # Create array for combining data
    arr_mcc = np.zeros((n_cases,n_runs,n_steps))
    arr_prf = np.zeros((n_cases,n_runs,n_steps))
    arr_lvl = np.zeros((n_cases,n_runs,n_steps,n_levels))

    # Iteratively open each file if it exists and import contents
    for ii in np.arange(n_cases):
        for jj in np.arange(n_runs):
            file = directory + f'case{ii:04}_run{jj:04}.npy'
            if os.path.exists(file):

                # Get data from file
                culture, org_prf, org_dem = load_exec003_data(file)

                # Write results to csv

                # Write results to array
                xy = culture[:,:,0] + culture[:,:,1]
                arr_mcc[ii,jj,:] = np.mean(xy,axis=1)
                arr_prf[ii,jj,:] = org_prf
                arr_lvl[ii,jj,:,:] = mean_culture_level(levels,xy)

    # Save results
    if test:
        loc = directory + 'test_results.npy'
    else:
        loc = directory + 'results.npy'
    with open(loc,'wb') as file:
        np.save(file,arr_mcc)
        np.save(file,arr_prf)
        np.save(file,arr_lvl)

    # Stop timer & print time
    t_stop = dt.datetime.now()
    print(t_stop - t_start)


def load_exec003_results(file_name):
    """Loads results from a saved MCC numpy results file."""

    with open(file_name, 'rb') as file:
        mcc = np.load(file)
        prf = np.load(file)
        lvl = np.load(file)

    return mcc, prf, lvl


#%% Call Script

if __name__ == '__main__':

    exec_num = 3
    name = f'culture_sim_exec{exec_num:03}'

    if exec_num == 1:

        if sys.platform.startswith('linux'):
            loc = '/gpfs1/home/j/m/jmeluso/culture_sim/data/' + name + '/'
            combine_exec001(loc,640,100,False)
        else:
            loc = 'data/' + name + '/'
            combine_exec001(loc,640,100)
            mcc, inc, prf, dem, lvl \
                = load_exec001_results(loc + 'test_results.npy')
            subset = lvl[639,1,:,:]

    elif exec_num == 2:

        if sys.platform.startswith('linux'):
            loc = '/gpfs1/home/j/m/jmeluso/culture_sim/data/' + name + '/'
            combine_exec002(loc,100,100,False)
        else:
            loc = 'data/' + name + '/'
            combine_exec002(loc,100,100)
            mcc, prf, lvl = load_exec002_results(loc + 'test_results.npy')

    else:

        if sys.platform.startswith('linux'):
            loc = '/gpfs1/home/j/m/jmeluso/culture_sim/data/' + name + '/'
            combine_exec003(loc,30,500,False)
        else:
            loc = 'data/' + name + '/'
            combine_exec003(loc,30,500)
            mcc, prf, lvl = load_exec003_results(loc + 'test_results.npy')




