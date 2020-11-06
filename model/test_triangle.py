# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:28:37 2020

@author: John Meluso
"""

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import Organization as og
from numpy.random import default_rng
import matplotlib.pyplot as plt

n =1000
phi = 15

#############################################################################

mu1 = 0.45
mu2 = 0.45

save1 = np.zeros((n,3))
for kk in np.arange(n):
    save1[kk,:] = og.triangle_beta(mu1,phi,mu2,phi)

fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')
ax.scatter(save1[:,0],save1[:,1],save1[:,2],s=1,c='blue')
ax.view_init(30,45)

#############################################################################

mu1 = 0.1
mu2 = 0.5

save1 = np.zeros((n,3))
for kk in np.arange(n):
    save1[kk,:] = og.triangle_beta(mu1,phi,mu2,phi)

ax = fig.add_subplot(222, projection='3d')
ax.scatter(save1[:,0],save1[:,1],save1[:,2],s=1,c='red')
ax.view_init(30,45)

#############################################################################

mu1 = 0.5
mu2 = 0.1

save1 = np.zeros((n,3))
for kk in np.arange(n):
    save1[kk,:] = og.triangle_beta(mu1,phi,mu2,phi)

ax = fig.add_subplot(223, projection='3d')
ax.scatter(save1[:,0],save1[:,1],save1[:,2],s=1,c='magenta')
ax.view_init(30,45)

#############################################################################

mu1 = 0.05
mu2 = 0.05

save1 = np.zeros((n,3))
for kk in np.arange(n):
    save1[kk,:] = og.triangle_beta(mu1,phi,mu2,phi)

ax = fig.add_subplot(224, projection='3d')
ax.scatter(save1[:,0],save1[:,1],save1[:,2],s=1,c='green')
ax.view_init(30,45)

#############################################################################

plt.draw()
plt.pause(.001)

