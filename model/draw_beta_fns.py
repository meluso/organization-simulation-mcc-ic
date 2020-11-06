# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 17:34:29 2020

@author: John Meluso
"""

import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import Organization as og


fig, ax = plt.subplots(1, 1)
mu = np.arange(0.1,1,0.2)
phi = 15

n = 1000
a = np.zeros(len(mu))
b = np.zeros(len(mu))
x = np.linspace(0.001,0.999,n)
ls = ['-','--','-.',':','-']
lw = [1,1,1,1,2]

for ii in np.arange(len(mu)):
    a, b = og.beta(mu[ii],phi)
    label = '%.1f' % mu[ii]
    ax.plot(x,beta.pdf(x,a,b),label=label,ls=ls[ii],lw=lw[ii])
ax.legend(loc='best', frameon=False)

plt.show()