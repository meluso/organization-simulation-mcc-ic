# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:34:43 2020

@author: John Meluso
"""

import numpy as np
import matplotlib.pyplot as plt
import data_manager as dm


#%% Exec001 Methods

def make_exec001_scen():
    """Specifies the conditions of each scenario. Indexing follows:
    [n_pops,pop_mode,pop1_culture,pop2_culture,pop_start,pop_hire]"""

    # Create the conditions for each scenario as a list of dicts
    scen = []

    # Scenario 1: One population, any culture is equally likely
    scen.append([1, 'uniform_2var'])

    # Scenario 2: One population, average culture
    scen.append([1,'beta_2var',{'min': 0.1,'max': 0.9}])

    # Scenario 3: Two populations, pop2 is more inclusive
    scen.append([
        2,
        'beta_2var',
        {'min': [0.5],'max': [0.9]},
        {'min': [0.1],'max': [0.9,'index_p1_culture']},
        {'min': [0.5],'max': [0.9]},
        'index_p1_start'
        ])

    # Scenario 4: Two populations, pop2 is more contest
    scen.append([
        2,
        'beta_2var',
        {'min': [0.1],'max': [0.9]},
        {'min': [0.5,'index_p1_culture'] ,'max': [0.9]},
        {'min': [0.5],'max': [0.9]},
        'index_p1_start'
        ])

    # Scenario 5: Two populations, more inclusive pop2 enters workforce
    scen.append([
        2,
        'beta_2var',
        {'min': [0.5],'max': [0.9]},
        {'min': [0.1],'max': [0.9,'index_p1_culture']},
        {'min': [0.6],'max': [0.9]},
        {'min': [0.5],'max': ['index_p1_start']}
        ])

    # Scenario 6: Two populations, more contest pop2 enters workforce
    scen.append([
        2,
        'beta_2var',
        {'min': [0.1],'max': [0.9]},
        {'min': [0.5,'index_p1_culture'] ,'max': [0.9]},
        {'min': [0.6],'max': [0.9]},
        {'min': [0.5],'max': ['index_p1_start']}
        ])

    return scen

def get_exec001_scen(cases):
    """Gets the cases corresponding to all scenarios. Indexing follows:
    [n_pops,pop_mode,pop1_culture,pop2_culture,pop_start,pop_hire]"""

    # Create return array cases and list for 6 scenarios
    y = np.zeros((len(cases),))
    z = [[],[],[],[],[],[],[]]

    # Create scenario indeces
    scen1 = 1
    scen2 = 2
    scen3 = 3
    scen4 = 4
    scen5 = 5
    scen6 = 6

    # Create simulation spec parameters
    index_pops = 0
    index_mode = 1
    index_p1_culture = 2
    index_p2_culture = 3
    index_p1_start = 4
    index_p1_hire = 5

    for ii in np.arange(len(cases)):

        # For Scenarios 1 & 2, have only 1 population
        if cases[ii][index_pops] == 1:
            if cases[ii][index_mode] == 'uniform_2var':
                y[ii] = scen1
                z[scen1].append(ii)
            else:
                y[ii] = scen2
                z[scen2].append(ii)

        # Scenarios 3-6 have 2 populations
        # If p2c < p1c, if start = hire -> 3, else 5
        # If p1c > p2c, if start = hire -> 4, else 6
        else:
            if cases[ii][index_p2_culture] < cases[ii][index_p1_culture]:
                if cases[ii][index_p1_start] == cases[ii][index_p1_hire]:
                    y[ii] = scen3
                    z[scen3].append(ii)
                else:
                    y[ii] = scen5
                    z[scen5].append(ii)
            else:
                if cases[ii][index_p1_start] == cases[ii][index_p1_hire]:
                    y[ii] = scen4
                    z[scen4].append(ii)
                else:
                    y[ii] = scen6
                    z[scen6].append(ii)

    return y, z


def load_exec001_conditions():
    """Loads all MCC data including cases, scenarios, and scenarios by case."""

    # Get complete list of cases run for MCC
    cases = dm.cases_exec001()

    # Get complete list of scenarios for MCC
    scen = make_exec001_scen()

    # Get the scenario corresponding to each case
    paired, indeces = get_exec001_scen(cases)

    return cases, scen, paired, indeces


#%% Exec002 Methods

def load_sim():
    """Loads all sim data including cases, scenarios, and scenarios by case."""




#%% Call Script

if __name__ == '__main__':

    exec_num = 2
    name = f'../data/culture_sim_exec{exec_num:03}/results.npy'

    if exec_num == 1:

        # Import results
        mcc, inc, prf, dem, lvl = dm.load_exec001_results(name)

        # Load MCC Cases & Scenarios
        cases, scen, paired, indeces = load_exec001_conditions()

    elif exec_num == 2:

        # Import results
        mcc, prf, lvl = dm.load_exec002_results(name)

        # Load Sim Cases & Scenarios
        cases = dm.cases_exec002()

    else:

        # Import results
        mcc, prf, lvl = dm.load_exec003_results(name)

        # Load Sim Cases & Scenarios
        cases = dm.cases_exec003()








