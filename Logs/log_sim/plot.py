#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 23:44:24 2021

@author: alex
"""

import sys
sys.path.append('../')

import pandas as pd
from pylab import *
import os
import matplotlib.pyplot as plt
# %matplotlib qt
datadir=sort(os.listdir(os.path.join(os.getcwd())))
datadir=np.setdiff1d(datadir,np.array(["plot.py"]))[-1]
data=pd.read_csv(os.path.join(os.getcwd(),datadir,'log.txt'))

f,axes=plt.subplots(nrows=6,ncols=6)

for i,key in enumerate([i for i in data.keys() if i!="t"]):
    
    cu_ax=axes.flatten()[i]
    cu_ax.plot(data['t'],data[key], label=key)
    cu_ax.grid()
    cu_ax.legend()

    if "acc" in key:
        cu_ax.set_ylim(-10,10)
    if "speed" in key:
        cu_ax.set_ylim(-50,50)
    if "pos" in key:
        cu_ax.set_ylim(-250,250)
    if "omegadot" in key:
        cu_ax.set_ylim(-100,100)
    if "omega" in key:
        cu_ax.set_ylim(-10,10)
    if "q" in key:
        cu_ax.set_ylim(-1,1)
    if "forces" in key:
        cu_ax.set_ylim(-450,450)
    if "torque" in key:
        cu_ax.set_ylim(-100,100)
    if "alpha" in key:
        cu_ax.set_ylim(-2,2)
    if "joystick" in key:
        cu_ax.set_ylim(min(data[key]), max(data[key]))

        
