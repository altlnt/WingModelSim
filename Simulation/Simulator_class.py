import numpy as np
from Gui_class import Viewer
from MoteurPhysique_class_bis import MoteurPhysique
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import time 


"Define class Simulator"

"This is the main class for the simulation"
"This acts as a framework to run both Viewer, which handles GUI and MoteurPhysique, which handles physics"




class Simulator():
    
    def __init__(self):
        
        self.viewer=Viewer()
        self.moteur_phy=MoteurPhysique()
        self.t0=-1
        
    def update(self):
        

        "Pull joystick input from viewer (viewer periodically updates the joystick input"
        "using pygame "

        joystick_input=np.array([self.viewer.joystick_L_horizontal,
        self.viewer.joystick_L_vertical,
        self.viewer.joystick_R_horizontal,
        self.viewer.joystick_R_vertical])        
        
        "update simulation with a time step of len dt"

        self.moteur_phy.update_sim(time.time()-self.t0, joystick_input)
        

        " T_init in moteur_phy is the time when joystick inputs become active"
        " before t=T_init, a step in thrust is applied "


        if time.time()-self.t0<self.moteur_phy.T_init:
            
            r,g,b,alpha=255,0,0,255
            self.viewer.g_translation.setColor((r,g,b,alpha))
            
        elif self.moteur_phy.takeoff==0:
            
            r,g,b,alpha=255,255,0,255
            self.viewer.g_translation.setColor((r,g,b,alpha))
            
        elif self.moteur_phy.takeoff>0:
            
            r,g,b,alpha=0,255,0,255
            self.viewer.g_translation.setColor((r,g,b,alpha))
            
        " the attitude and pos information pulled from MoteurPhysique to update GUI camera "

        self.viewer.target_q=self.moteur_phy.q
        self.viewer.target_pos=self.moteur_phy.pos

        self.viewer.update_translation()
        self.viewer.update_joysticks()        
        
    def launch_sim(self):
        "popus up the GUI and starts the callback"
        self.viewer.w_translation.show()
        self.viewer.mw.show()

        self.t0=time.time()
        self.viewer.t = QtCore.QTimer()
        self.viewer.t.timeout.connect(self.update)
        self.viewer.t.start(50)
        pg.mkQApp().exec_()        
        
        return
