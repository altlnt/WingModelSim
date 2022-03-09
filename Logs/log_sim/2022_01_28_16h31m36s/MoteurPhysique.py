#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 00:36:07 2021
@author: alex
"""

import sys
sys.path.append('../')
import numpy as np
import os
import time
from datetime import datetime
import transforms3d as tf3d
import dill as dill
import json
from collections import OrderedDict

dill.settings['recurse'] = True


class MoteurPhysique():
    def __init__(self,called_from_opti=False,T_init=1.0):
        
        
        self.save_path_base=os.path.join("../Logs/log_sim",datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss"))

        if not called_from_opti:
            os.makedirs(self.save_path_base)
 
        
        # Miscellaneous
        self.data_save_path=self.save_path_base
        self.last_t=0.0
        self.T_init=T_init
        # Body state
        
        #   Translation
        self.forces,self.torque=np.zeros(3),np.zeros(3)
        self.grad_forces, self.grad_torque = np.zeros((3,10)), np.zeros((3,10))
        self.acc=np.zeros(3)
        self.speed=np.array([0.001,0,0])
        self.pos=np.zeros(3)
        
        #   Rotation
        self.omegadot=np.zeros(3)
        self.omega=np.array([0,0,0])
        self.q=np.array([1,0,0,0])
        self.R=tf3d.quaternions.quat2mat(self.q) 
        self.takeoff =0
        
        self.Effort_function = dill.load(open('../Simulation/function_moteur_physique','rb'))
        self.joystick_input = [0,0,0,0,0]
        self.joystick_input_log= [0,0,0,0,0]
        self.y_data=np.array([0,0,0,0,0,0])
        self.W = np.eye(6)
        
        ####### Dictionnaire des paramètres du monde 
        self.Dict_world     = {"g"    : np.array([0,0,9.81]),                    \
                               "mode" : [1,1,1,1]}   # Permet de bloquer les commandes (0 commande bloquer, 1 commande active)
        ####### Dictionnaire des paramètres pour l'optimisation
        self.Dict_variables = {"wind_X" :0,  \
                               "wind_Y" :0,  \
                               "wind_Z" :0,  \
                               "masse": 8.5 , \
                               "inertie": np.diag([1.38,0.84,2.17]),\
                               "aire" : [0.62*0.262* 1.292 * 0.5, 0.62*0.262* 1.292 * 0.5, 0.34*0.01* 1.292 * 0.5, 0.34*0.1* 1.292 * 0.5, 1.08*0.31* 1.292 * 0.5],\
                               "cp_list": [np.array([-0.013,0.475,-0.040],       dtype=np.float).flatten(), \
                                           np.array([-0.013,-0.475,-0.040],      dtype=np.float).flatten(), \
                                           np.array([-1.006,0.85,-0.134],    dtype=np.float).flatten(),\
                                           np.array([-1.006,-0.85,-0.134],   dtype=np.float).flatten(),\
                                           np.array([0.021,0,-0.064],          dtype=np.float).flatten()],
                               "alpha0" : 0*np.array([0.07,0.07,0,0,0.07]),\
                               "alpha_stall" : 5 ,                     \
                               "largeur_stall" : 15.0*np.pi/180,                  \
                               "cd0sa" : 0.05,\
                               "cd0fp" : 0.00,\
                               "cd1sa" : 4.55, \
                               "cl1sa" : 5, \
                               "cd1fp" : 0.0, \
                               "coeff_drag_shift": 0.0, \
                               "coeff_lift_shift": 0.0, \
                               "coeff_lift_gain": 2.5,\
                               "Ct": 2e-4, \
                               "Cq": 0, \
                               "Ch": 0,\
                               "rotor_moy_speed":180}
            
        self.moy_rotor_speed = self.Dict_variables["rotor_moy_speed"]

            
        # self.Dict_variables = OrderedDict(sorted(self.Dict_variables.items(), key=lambda t: t[0]))
            
        # Dictionnaire des états pour la jacobienne
        if not dill.load(open('../Simulation/function_moteur_physique','rb'))[-1]==None:
            self.Theta=dill.load(open('../Simulation/function_moteur_physique','rb'))[-1]
        else:
            print("Attention aux chargement des params pour l'identif")
            self.Theta=['alpha0',
                         'cd1sa',
                          'cl1sa',
                          'cd0sa',
                          'coeff_drag_shift',
                          'coeff_lift_shift',
                          'coeff_lift_gain']

        
            
        # Dictionnaires des états
        self.Dict_etats     = {"position" : self.pos,    \
                               "vitesse" : self.speed,   \
                               "acceleration" : self.acc,\
                               "orientation" : self.q,   \
                               "vitesse_angulaire" : self.omega, \
                               "accel_angulaire" : self.omegadot,\
                               "alpha" : [0.0]*5}
            
        self.Dict_Var_Effort = {"Omega" :self.omega,\
                                "speed": self.speed, \
                                "Cd_list": np.array([0,0,0,0,0]), \
                                "Cl_list": np.array([0,0,0,0,0]), \
                                }
        
        self.Dict_Commande = {"delta" : 0,\
                              "rotor_speed" : self.moy_rotor_speed }
 
    
        self.SaveDict={} 
        self.called_from_opti=called_from_opti
        if not called_from_opti:
            for dic in [self.Dict_world,self.Dict_variables]:
                keys=dic.keys() 
                for key in keys:
                    self.SaveDict[key]=np.array(dic[key]).tolist()
            with open(os.path.join(self.save_path_base,'params.json'), 'w') as fp:
                json.dump(self.SaveDict, fp)
            with open(os.path.join(self.save_path_base,'MoteurPhysique.py'), 'w') as mp:
                print("saved_ moteur")
                mp.write(open("MoteurPhysique_class.py").read())

            print(self.data_save_path)
    
    def orthonormalize(self,R_i):
        R=R_i
        R[:,0]/=np.linalg.norm(R[:,0])
        R[:,1]=R[:,1]-np.dot(R[:,0].flatten(),R[:,1].reshape((3,1)))*R[:,0]
        R[:,1]/=np.linalg.norm(R[:,1])
        R[:,2]=np.cross(R[:,0],R[:,1])
        R[:,2]/=np.linalg.norm(R[:,2])
        return R

    def Rotation(self,R,angle):
        c, s = np.cos(angle*np.pi/180), np.sin(angle*np.pi/180)
        r = np.array(( (1,0, 0), (0,c, s),(0,-s, c)) , dtype=np.float)
        return R @ r
    
    def Init(self,joystick_input,t):
        self.joystick_input_log= joystick_input
        self.joystick_input = joystick_input
        self.moy_rotor_speed = self.Dict_variables["rotor_moy_speed"]

        for q,i in enumerate(joystick_input):      # Ajout d'une zone morte dans les commandes 
            if abs(i)<40 :
                    self.joystick_input[q] = 0

            elif q==3 :
                if joystick_input[q]/250<-0.70:
                    self.joystick_input[q]=0
                else:
                    self.joystick_input[q] = joystick_input[q]/250 *  self.moy_rotor_speed
            else:
                self.joystick_input[q] = joystick_input[q] * 15 *np.pi/180 / 250 
        
        for j,p in enumerate(self.Dict_world["mode"]):
            if p==0:
                self.joystick_input[j] = 0
                
           
        # Mise à niveau des commandes pour etre entre -15 et 15 degrés 
         # (l'input est entre -250 et 250 initialement)
        self.Dict_Commande["delta"] = np.array([self.joystick_input[0], -self.joystick_input[0], \
                                                (self.joystick_input[1] - self.joystick_input[2]) \
                                                , (self.joystick_input[1] + self.joystick_input[2]) , 0])
            
        for l,m in enumerate(self.Dict_Commande["delta"]):
            if m>15:
                self.Dict_Commande["delta"][l]=15
        
        self.Dict_Commande["rotor_speed"] =  self.moy_rotor_speed + self.joystick_input[3]
                                                                     
        R_list         = [self.R, self.R, self.Rotation(self.R, 45), self.Rotation(self.R,-45), self.R]
        v_W            = np.array([self.Dict_variables["wind_X"],self.Dict_variables["wind_Y"],self.Dict_variables["wind_Z"]])
        frontward_Body = np.transpose(np.array([[1,0,0]]))
        A_list         = self.Dict_variables["aire"]
        alpha_0_list   = self.Dict_variables["alpha0"]
        alpha_s        = self.Dict_variables["alpha_stall"]
        delta_s        = self.Dict_variables["largeur_stall"]  
        cp_list        = self.Dict_variables['cp_list']
        cd1sa = self.Dict_variables["cd1sa"]            
        cl1sa = self.Dict_variables["cl1sa"]
        cd0sa = self.Dict_variables["cd0sa"]
        cd1fp = self.Dict_variables["cd1fp"]
        cd0fp = self.Dict_variables["cd0fp"]
        k0    = self.Dict_variables["coeff_drag_shift"]
        k1    = self.Dict_variables["coeff_lift_shift"]
        k2    = self.Dict_variables["coeff_lift_gain"]
        
        return R_list,v_W,frontward_Body, A_list, alpha_0_list, alpha_s, delta_s, cp_list, cd1sa, cl1sa, cd0sa, cd1fp, cd0fp, k0, k1, k2
    
    def EulerAngle(self, q):
        # Calcul les angles roll, pitch, yaw en fonction du quaternion, utiliser uniquement pour le plot
        sinr_cosp = 2 * (q[0] * q[1] + q[2] * q[3])
        cosr_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (q[0] * q[2] - q[3] * q[1])
        if (abs(sinp) >= 1):
            pitch = np.sign(np.pi/ 2, sinp) 
        else:
            pitch = np.arcsin(sinp)
        siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2]);
        cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll,pitch,yaw])
    
    def compute_dynamics(self,joystick_input,t, compute_gradF=None):
        # Ouverture du fichier de fonction, on obtient une liste de fonction obtenu avec le jupyter lab.
            
        T_init=self.T_init    # Temps pendant laquelle les forces ne s'appliquent pas sur le drone

        R_list,v_W,frontward_Body, A_list, alpha_0_list, alpha_s, delta_s, cp_list,\
            cd1sa, cl1sa, cd0sa, cd1fp, cd0fp, k0, k1, k2 = self.Init(joystick_input,t)
            
        if compute_gradF==None:
            if (t)<T_init:
                self.forces= np.array([0,0,0]) 
                self.torque= np.array([0,0,0])
                if not self.called_from_opti:
                    print("Début des commandes dans :", T_init-t)
            else: 
                alpha_list=[0,0,0,0,0]
                
                for p, cp in enumerate(cp_list) :          # Cette boucle calcul les coefs aéro pour chaque surface 
                    VelinLDPlane  = self.Effort_function[0](self.omega, cp, self.speed.flatten(), v_W, R_list[p].flatten())
                    dragDirection = self.Effort_function[1](self.omega, cp, self.speed.flatten(), v_W, R_list[p].flatten())
                    liftDirection = self.Effort_function[2](self.omega, cp, self.speed.flatten(), v_W, R_list[p].flatten())
                    alpha_list[p] = self.Effort_function[3](dragDirection, liftDirection, frontward_Body, VelinLDPlane)

                self.Dict_etats['alpha'] = alpha_list

                forces, torque = self.Effort_function[4](A_list, self.omega, self.R.flatten(), self.speed.flatten(),\
                                                  v_W, cp_list,alpha_list, alpha_0_list,\
                                                   alpha_s, self.Dict_Commande["delta"], \
                                                   delta_s, cl1sa, cd1fp, k0, k1, k2, cd0fp, \
                                                   cd0sa, cd1sa, \
                                                  self.Dict_variables["Ct"], self.Dict_variables["Cq"], \
                                                  self.Dict_variables["Ch"],self.Dict_Commande["rotor_speed"])

                self.forces= self.R @ np.transpose(forces.flatten()) +  self.Dict_variables["masse"] *self.Dict_world["g"]
                self.torque = np.transpose(torque).flatten()  
                if self.takeoff==0:
                    self.forces[2]=min(self.forces[2],0)
                    
                    
################## Calcul du gradient #################
        else:
            alpha_list=[0,0,0,0,0]
            for p, cp in enumerate(cp_list) :          
                VelinLDPlane   = self.Effort_function[0](self.omega, cp, self.speed.flatten(), v_W, R_list[p].flatten())
                dragDirection  = self.Effort_function[1](self.omega, cp, self.speed.flatten(), v_W, R_list[p].flatten())
                liftDirection  = self.Effort_function[2](self.omega, cp, self.speed.flatten(), v_W, R_list[p].flatten())
                alpha_list[p] = self.Effort_function[3](dragDirection, liftDirection, frontward_Body, VelinLDPlane)
            
            #######Calcul du gradient du cout avec une fonction symbolique qui reprend tout le calcul précédent

            self.grad_cout = self.Effort_function[9](self.y_data.flatten(), A_list, self.omega, self.R.flatten(), self.speed.flatten(),\
                                              v_W, cp_list,alpha_list, alpha_0_list,\
                                               alpha_s, self.Dict_Commande["delta"], \
                                               delta_s, cl1sa, cd1fp, k0, k1, k2, cd0fp, \
                                               cd0sa, cd1sa, \
                                              self.Dict_variables["Ct"], self.Dict_variables["Cq"], \
                                              self.Dict_variables["Ch"],self.Dict_Commande["rotor_speed"],\
                                              self.Dict_world["g"].flatten(),self.Dict_variables["masse"],\
                                              self.W[0,0], self.W[1,1], self.W[2,2],self.W[3,3], self.W[4,4], self.W[5,5])
 
     
    def compute_cost(self,joystick_input,t):
        #### Calcul du cout (Somme des erreurs)

        R_list,v_W,frontward_Body, A_list, alpha_0_list, alpha_s, delta_s, cp_list,\
            cd1sa, cl1sa, cd0sa, cd1fp, cd0fp, k0, k1, k2 = self.Init(joystick_input,t) 
        alpha_list=[0,0,0,0,0]
        for p, cp in enumerate(cp_list) :          # Cette boucle calcul les coefs aéro pour chaque surface 
            VelinLDPlane   = self.Effort_function[0](self.omega, cp, self.speed.flatten(), v_W, R_list[p].flatten())
            dragDirection  = self.Effort_function[1](self.omega, cp, self.speed.flatten(), v_W, R_list[p].flatten())
            liftDirection  = self.Effort_function[2](self.omega, cp, self.speed.flatten(), v_W, R_list[p].flatten())
            alpha_list[p] = self.Effort_function[3](dragDirection, liftDirection, frontward_Body, VelinLDPlane)            

        RMS_forces = self.Effort_function[6](self.y_data, A_list, self.omega, self.R.flatten(), self.speed.flatten(),\
                                              v_W, cp_list,alpha_list, alpha_0_list,\
                                               alpha_s, self.Dict_Commande["delta"], \
                                               delta_s, cl1sa, cd1fp, k0, k1, k2, cd0fp, \
                                               cd0sa, cd1sa, \
                                              self.Dict_variables["Ct"], self.Dict_variables["Cq"], \
                                              self.Dict_variables["Ch"],self.Dict_Commande["rotor_speed"],\
                                              self.Dict_world["g"].flatten(),self.Dict_variables["masse"],self.W[0,0], self.W[1,1], self.W[2,2] )
            
        RMS_torque = self.Effort_function[7](self.y_data, A_list, self.omega, self.R.flatten(), self.speed.flatten(),\
                                              v_W, cp_list,alpha_list, alpha_0_list,\
                                               alpha_s, self.Dict_Commande["delta"], \
                                               delta_s, cl1sa, cd1fp, k0, k1, k2, cd0fp, \
                                               cd0sa, cd1sa, \
                                              self.Dict_variables["Ct"], self.Dict_variables["Cq"], \
                                              self.Dict_variables["Ch"],self.Dict_Commande["rotor_speed"],\
                                              self.Dict_world["g"].flatten(),self.Dict_variables["masse"], self.W[3,3], self.W[4,4], self.W[5,5])
            
        return RMS_forces+RMS_torque, RMS_forces, RMS_torque
    
    
    def update_state(self,dt):
        
        "update omega"
        
        J=self.Dict_variables['inertie']
        J_inv=np.linalg.inv(J)
        m=np.ones(3) * self.Dict_variables['masse']
        
        new_omegadot=-np.cross(self.omega.T,np.matmul(J,self.omega.reshape((3,1))).flatten())
        new_omegadot=J_inv @ np.transpose(new_omegadot+self.torque)
        new_omega=self.omega+new_omegadot.flatten()*dt        

        self.omegadot=new_omegadot
        self.omega=new_omega
        
        if abs(self.pos[2])<0.001 and self.takeoff==0:
            self.omega=np.array([0,max(self.omega[1],0),self.omega[2]]) 
        elif abs(self.pos[2])>0.001 and self.takeoff==0:
            print("Décollage effectué")
            self.takeoff = 1 
        
        "update q"
        
        qs,qv=self.q[0],self.q[1:]
        dqs=-0.5*np.dot(qv,self.omega)
        dqv=0.5*(qs*self.omega+np.cross(qv.T,self.omega.T).flatten())   
        dq=np.r_[dqs,dqv]
        new_q=self.q+dq*dt        
            
        R=tf3d.quaternions.quat2mat(new_q/tf3d.quaternions.qnorm(new_q))
        self.R=self.orthonormalize(R)
        self.q=tf3d.quaternions.mat2quat(R)    
        #print(tf3d.quaternions.qnorm(self.q))
        "update forces"
                
        self.acc=self.forces/m
        self.speed=self.speed+self.acc*dt
        self.pos=self.pos+self.speed*dt        

    def log_state(self):
        
        keys=['t','acc[0]','acc[1]','acc[2]',
              'speed[0]','speed[1]','speed[2]',
              'pos[0]','pos[1]','pos[2]',
              'omegadot[0]','omegadot[1]','omegadot[2]',
              'omega[0]','omega[1]','omega[2]',
              'q[0]','q[1]','q[2]','q[3]',
              'forces[0]','forces[1]','forces[2]',
              'torque[0]','torque[1]','torque[2]',
              'alpha[0]','alpha[1]','alpha[2]','alpha[3]','alpha[4]',
              'joystick[0]','joystick[1]','joystick[2]','joystick[3]',
              'delta[0]','delta[1]','delta[2]','delta[3]','rot_speed','takeoff']
        
        t=self.last_t
        acc=self.acc
        speed=self.speed
        pos=self.pos
        omegadot=self.omegadot
        omega=self.omega
        q=self.q
        forces=self.forces
        torque=self.torque
        alpha=self.Dict_etats['alpha']
        euler = self.EulerAngle(q) * 180/np.pi
        joystick_input = self.joystick_input_log
        delta = self.Dict_Commande["delta"]
        rot_speed=self.Dict_Commande["rotor_speed"]
        TK=self.takeoff

        if 'log.txt' not in os.listdir(self.data_save_path):
            print("Here: Init")
            first_line=""
            for j,key in enumerate(keys):
                if j!=0:
                    first_line=first_line+","
                first_line=first_line+key
                
            first_line=first_line+"\n"
            with open(os.path.join(self.data_save_path,"log.txt"),'a') as f:
                f.write(first_line)
        
        scope=locals()
        list_to_write=[t,acc[0],acc[1],acc[2],
              speed[0],speed[1],speed[2],
              pos[0],pos[1],pos[2],
              omegadot[0],omegadot[1],omegadot[2],
              omega[0],omega[1],omega[2],
              q[0],q[1],q[2],q[3],
              forces[0],forces[1],forces[2],
              torque[0],torque[1],torque[2], 
              alpha[0],alpha[1],alpha[2],alpha[3],alpha[4],
              joystick_input[0], joystick_input[1],
              joystick_input[2], joystick_input[3],
              delta[0],delta[1],delta[2],delta[3],rot_speed,
              TK]
        
        
        
        line_to_write=''
        for j,element in enumerate(list_to_write):
            if j!=0:
                line_to_write=line_to_write+","            
            line_to_write=line_to_write+str(element)
            
        line_to_write=line_to_write+"\n"  
        
        with open(os.path.join(self.data_save_path,"log.txt"),'a+') as f:
            #print("Here: data")
            f.write(line_to_write)        
            
    def update_sim(self,t,joystick_input):
        
        dt=t-self.last_t
        self.last_t=t
        self.compute_dynamics(joystick_input,self.last_t)
        self.update_state(dt)
        self.log_state()
        
        return