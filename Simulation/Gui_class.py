import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
import socket
import pygame
import transforms3d as tf3d 
from PIL import Image 

"Viewer class"
"a symple pyqtgraph based gui to visualize the drone's motion"

class Viewer:
    def __init__(self):
        pygame.joystick.init()
        pygame.init()
        self.app = pg.mkQApp()
        ####################################################################
        
        #           Initialisation du widget de visualisation de la rotation
        "3D rotation"       
        
        self.target_q=np.array([1.0,0.0,0.0,0.0])
        self.target_R=tf3d.quaternions.quat2mat(self.target_q)

        #  Création de la géométrie (on pourrait mettre un mesh de drone pour être plus "réaliste")
        
        
        self.verts = np.array([
            [1, 0, 0],
            [0, 1.5, 0],
            [0,-1.5,0],
            [-0.5,0,0],
            [-0.75,-0.5,0],
            [-0.75, 0.5, 0],
            [-0.75,0,-0.5],
            [-0.75,0,0]
        ])
        self.verts*=1.5
        self.faces = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [3, 6, 7]
        
        ])
        self.colors = np.array([
            [1, 0, 0, 100],
            [0, 1, 0, 100],
            [0, 0, 1, 100]])
        
        self.m1 = gl.GLMeshItem(vertexes=self.verts, faces=self.faces, faceColors=self.colors, smooth=False)
        self.m1.translate(0, 0, 0)
        self.m1.setGLOptions('additive')
        
        
        self.phase=0
        ####################################################################
        
        #           Initialisation du widget de visualisation de la translation
        
        "2d translation"
        img = Image.open('background.png') 
        img=np.asarray(img) 
        v1 = gl.GLImageItem(img) 
        v1.translate(-img.shape[0]/2, -img.shape[1]/2, 0) 

        self.w_translation = gl.GLViewWidget()
        self.w_translation.opts['distance'] = 20
        self.w_translation.setWindowTitle('Translation')
        self.w_translation.addItem(v1)
        
        self.g_translation = gl.GLGridItem()
        self.g_translation.setSize(100,100,100)
        self.w_translation.addItem(self.g_translation)
        
        self.pos = [[0,0,0]]
        self.target_pos=None
        
        self.Msize=0.001
        self.Mcolor=(1.0,1.0,0.0,1.0)
        self.size=0.1
        self.color=(1.0,0.0,0.0,0.2)
        
        self.cols=[self.Mcolor]
        self.sizes=[self.Msize]
        
        
        self.w_translation.addItem(self.m1)


        #                   Callback update de l'affichage de la translation 
        
        ####################################################################
        
        #           Initialisation du widget de gestion du joystick
        
        self.mw = QtGui.QMainWindow()
        self.mw.resize(150,50)
        self.mw.setWindowTitle('JoystickButton')
        self.cw = QtGui.QWidget()
        self.mw.setCentralWidget(self.cw)
        self.layout = QtGui.QGridLayout()
        self.cw.setLayout(self.layout)

        self.jb_l = pg.JoystickButton()
        self.jb_l.setFixedWidth(50)
        self.jb_l.setFixedHeight(50)
        
        
        self.jb_r = pg.JoystickButton()
        self.jb_r.setFixedWidth(50)
        self.jb_r.setFixedHeight(50)
        
        self.layout.addWidget( self.jb_l, 0, 0)
        self.layout.addWidget( self.jb_r, 0, 1)

        

        " paramètres de gestion du joystick "

        self.joystick_number=0
        
        self.JS=pygame.joystick.Joystick(self.joystick_number)        
        self.JS.init()

        self.joystick_L_horizontal_number=0
        self.joystick_L_vertical_number=1
        self.joystick_R_horizontal_number=3
        self.joystick_R_vertical_number=4



        LH=self.JS.get_axis(self.joystick_L_horizontal_number)
        LV=self.JS.get_axis(self.joystick_L_vertical_number)
        RH=self.JS.get_axis(self.joystick_R_horizontal_number)
        RV=self.JS.get_axis(self.joystick_R_vertical_number)
        
        # l'input du joystick doit être rescalé

        
        self.joystick_L_horizontal = LH/4 #left -1 / right 1
        self.joystick_L_vertical = -LV/4  #up -1 / down 1
        self.joystick_R_horizontal = RH/4#left -1 / right 1
        self.joystick_R_vertical = -RV/4  #up -1 / down 1
        
            
        self.joystick_L_horizontal*=1e3
        self.joystick_L_vertical*=1e3
        self.joystick_R_horizontal*=1e3
        self.joystick_R_vertical*=1e3
    
        self.jb_l.setState(self.joystick_L_horizontal,self.joystick_L_vertical)
        self.jb_r.setState(self.joystick_R_horizontal,self.joystick_R_vertical)
        
        return
            
    def update_translation(self,new_position=None):
        
        " cette méthode est utilsée par la classe simulateur, dans Simulator_class.py, qui instancie la classe Viewer "
        " elle sert à updater la visualisation"

        self.cols[-1]=self.color
        self.cols.append(self.Mcolor)
        self.sizes[-1]=self.size
        self.sizes.append(self.Msize)
    
    
    
        ################### récupérer la position du modèle et la feeder ici
        self.new_pos=np.array([self.target_pos[0],self.target_pos[1],-self.target_pos[2]]) 
        ###################        
        self.pos.append(self.new_pos)

        self.target_R=tf3d.quaternions.quat2mat(self.target_q)
        TR=np.eye(4)
        TR[:3,:3]=np.diag([1.0,1.0,-1.0])@self.target_R
        TR[:-1,-1]=self.new_pos
        
        self.m1.setTransform(TR)
        self.m1.update()    
        self.w_translation.opts['rotationMethod'] = "quaternion"
        self.w_translation.setCameraPosition(pos=QtGui.QVector3D(self.new_pos[0], \
                                                                 self.new_pos[1], \
                                                                 self.new_pos[2]))
                                              # rotation=QtGui.QVector4D(self.target_q[0],self.target_q[1]\
                                              #                          ,self.target_q[2],self.target_q[3]))
    
    def update_joysticks(self):

        "cette méthode est appelée par la callback, elle permet de récupérer les inputs du joystick"

        pygame.event.get()
        self.JS=pygame.joystick.Joystick(self.joystick_number)
        self.JS.init()

        # print(self.JS.get_id())
        
        LH=self.JS.get_axis(self.joystick_L_horizontal_number)
        LV=self.JS.get_axis(self.joystick_L_vertical_number)
        RH=self.JS.get_axis(self.joystick_R_horizontal_number)
        RV=self.JS.get_axis(self.joystick_R_vertical_number)
        

        
        self.joystick_L_horizontal = LH/4 #left -1 / right 1
        self.joystick_L_vertical = -LV/4  #up -1 / down 1
        self.joystick_R_horizontal = RH/4#left -1 / right 1
        self.joystick_R_vertical = -RV/4  #up -1 / down 1
        
                
        self.joystick_L_horizontal*=1e3
        self.joystick_L_vertical*=1e3
        self.joystick_R_horizontal*=1e3
        self.joystick_R_vertical*=1e3
    
        self.jb_l.setState(self.joystick_L_horizontal,self.joystick_L_vertical)
        self.jb_r.setState(self.joystick_R_horizontal,self.joystick_R_vertical)        
        
        return
        
    def update(self):

        "wrapper pour les méthodes d'update définies ci-dessus"
        self.update_rot()
        self.update_translation()
        self.update_joysticks()
        self.phase -= 0.1
        return
    
    def launch(self):

        "pour démarrer la callback de la fenêtre"

        self.w_rot.show()
        self.w_translation.show()
        self.mw.show()

        self.t = QtCore.QTimer()
        self.t.timeout.connect(self.update)
        self.t.start(25)
        pg.mkQApp().exec_()
        return
    
