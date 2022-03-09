#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 10:10:24 2022

@author: alex
"""

import numpy as np
import transforms3d as tf3d
import matplotlib.pyplot as plt
import os
import pandas as pd


log_path="datasets/sa/log.txt"

lpl=log_path.split('/')
lpl[-2]=lpl[-2]+'_noisy'

os.makedirs(os.path.join(*tuple(lpl[:-1])),exist_ok=True)

noisy_log_path=os.path.join(*tuple(lpl))

raw_data=pd.read_csv(log_path)

prep_data=raw_data.drop(columns=[i for i in raw_data.keys() if (("forces" in i ) or ('pos' in i) ) ])

std_acc = 1.0
std_speed = 0.1
std_omega = 0.08

f,axes=plt.subplots(3,3)

"add noise on acc,speed,omega"
for i in range(3):
    for j,s,k in zip(range(3),[std_speed,std_acc,std_omega],
                   ["speed","acc","omega"]):
        
        axes[i,j].plot(prep_data['%s[%i]'%(k,i)],label=k)
        prep_data['%s[%i]'%(k,i)]=prep_data['%s[%i]'%(k,i)] \
            + np.random.normal(0,s,size=len(prep_data))
        axes[i,j].plot(prep_data['%s[%i]'%(k,i)],label=k)
        axes[i,j].legend()

std_angle = 2.0*np.pi/180

R_list=[tf3d.quaternions.quat2mat([i,j,k,l]) for i,j,k,l in zip(prep_data['q[0]'],
                                                                prep_data['q[1]'],
                                                                prep_data['q[2]'],
                                                                prep_data['q[3]'])]

def orthonormalize(R_i):
    R=R_i
    R[:,0]/=np.linalg.norm(R[:,0])
    R[:,1]=R[:,1]-np.dot(R[:,0].flatten(),R[:,1].reshape((3,1)))*R[:,0]
    R[:,1]/=np.linalg.norm(R[:,1])
    R[:,2]=np.cross(R[:,0],R[:,1])
    R[:,2]/=np.linalg.norm(R[:,2])
    return R
    
def skew(x):
    return np.array([[0,-x[2],x[1]],
                      [x[2],0,-x[0]],
                      [-x[1],x[0],0]])
        
def pipeline(R):
    noise_angle= np.random.normal(0,std_angle,size=3)
    new_R = R@(np.eye(3)+skew(noise_angle))        
    new_R = orthonormalize(new_R)
    return new_R

new_R_list=np.array(list(map(pipeline,R_list)))
q_list=[np.array([i,j,k,l]) for i,j,k,l in zip(prep_data['q[0]'],
                                                prep_data['q[1]'],
                                                prep_data['q[2]'],
                                                prep_data['q[3]'])]
new_q_list=np.array(list(map(tf3d.quaternions.mat2quat,new_R_list)))

plt.figure()
plt.plot(q_list,"--")
plt.plot(new_q_list)

prep_data['q[0]'],\
prep_data['q[1]'],\
prep_data['q[2]'],\
prep_data['q[3]'] = new_q_list.T

prep_data.to_csv(noisy_log_path,index=False)
