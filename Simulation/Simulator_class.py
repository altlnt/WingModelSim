#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 21:33:53 2021

@author: alex
"""
import numpy as np
from Gui_class import Viewer
from MoteurPhysique_class_bis import MoteurPhysique
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import time 

class Simulator():
    
    def __init__(self):
        
        self.viewer=Viewer()
        self.moteur_phy=MoteurPhysique()
        self.t0=-1
        
    def update(self):
        
        joystick_input=np.array([self.viewer.joystick_L_horizontal,
        self.viewer.joystick_L_vertical,
        self.viewer.joystick_R_horizontal,
        self.viewer.joystick_R_vertical])        
        
        
        self.moteur_phy.update_sim(time.time()-self.t0, joystick_input)
        
        if time.time()-self.t0<self.moteur_phy.T_init:
            
            r,g,b,alpha=255,0,0,255
            self.viewer.g_translation.setColor((r,g,b,alpha))
            
        elif self.moteur_phy.takeoff==0:
            
            r,g,b,alpha=255,255,0,255
            self.viewer.g_translation.setColor((r,g,b,alpha))
            
        elif self.moteur_phy.takeoff>0:
            
            r,g,b,alpha=0,255,0,255
            self.viewer.g_translation.setColor((r,g,b,alpha))
            
        self.viewer.target_q=self.moteur_phy.q
        self.viewer.target_pos=self.moteur_phy.pos

        self.viewer.update_translation()
        self.viewer.update_joysticks()        
        
    def launch_sim(self):
        
        self.viewer.w_translation.show()
        self.viewer.mw.show()

        self.t0=time.time()
        self.viewer.t = QtCore.QTimer()
        self.viewer.t.timeout.connect(self.update)
        self.viewer.t.start(50)
        pg.mkQApp().exec_()        
        
        return

# S=Simulator()
# S.launch_sim()