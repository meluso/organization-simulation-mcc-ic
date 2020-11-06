# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:07:45 2020

@author: John Meluso
"""

import Organization as og
import numpy as np
import pickle


def save_mcc(results,file_name):
    """Saves results data from test for the MCC simulation trials."""

    mcc = np.mean(results.socialization[:,:,0] \
                  + results.socialization[:,:,1],axis=1)
    inc = np.mean(results.socialization[:,:,2],axis=1)
    prf = results.performance_org
    dem = results.demographics

    with open(file_name,'wb') as file:
        np.save(file,mcc)
        np.save(file,inc)
        np.save(file,prf)
        np.save(file,dem)


def load_mcc(file_name):
    """Loads results from a saved MCC numpy trial file."""

    with open(file_name, 'rb') as file:
        mcc = np.load(file)
        inc = np.load(file)
        prf = np.load(file)
        dem = np.load(file)

    return mcc, inc, prf, dem


def load_trials():
    """Loads the trial parameter list."""
    return pickle.load(open('trials.pickle','rb'))


def generate_levels():
    """Generates levels for a (4,5) clique tree."""
    level = np.zeros((781,1))
    level[0] = 5
    level[1:6] = 4
    level[6:31] = 3
    level[31:156] = 2
    level[156:781] = 1
    return level