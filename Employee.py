# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:41:20 2020

@author: John Meluso
"""

import numpy as np

class Employee(object):
    '''Defines a class Employee which contains the properties of an employee in
    a specified organization.'''

    def __init__(self,mode=0):

        if mode == 0:
            # Initialize values for random start
