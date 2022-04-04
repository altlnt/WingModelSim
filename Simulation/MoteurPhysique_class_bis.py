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
 
        "defining placeholders"
        # Miscellaneous
        self.last_t=0.0 #last timestep is kept in memory
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
        
        self.joystick_input = [0,0,0,0,0]
        self.joystick_input_log= [0,0,0,0,0]
        self.y_data=np.array([0,0,0,0,0,0])
        
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
                               "cp_list": [np.array([-0.0,0.5,0.0],       dtype=np.float).flatten(), \
                                           np.array([-0.0,-0.5,0.0],      dtype=np.float).flatten(), \
                                           np.array([-1.00,0.85,-0.134],    dtype=np.float).flatten(),\
                                           np.array([-1.00,-0.85,-0.134],   dtype=np.float).flatten(),\
                                           np.array([0.0,0,0.0],          dtype=np.float).flatten()],
                               "alpha0" : 0*np.array([0.07,0.07,0,0,0.07]),\
                               "alpha_stall" : 15.0*np.pi/180 ,                     \
                               "largeur_stall" : 15.0*np.pi/180,                  \
                               "cd0sa" : 0.025,\
                               "cd1sa" : 4.55, \
                               "cl1sa" : 5, \
                               "cd0fp" : 0.025,\
                               "cd1fp" : 2.0, \
                               "cl1fp" : 3.65, \
                               "coeff_drag_shift": 0.0, \
                               "coeff_lift_shift": 0.0, \
                               "coeff_lift_gain": 2.5,\
                               "Ct": 2*2e-4, \
                               "Cq": 0, \
                               "Ch": 0,\
                               "rotor_moy_speed":150}
            
        self.Aire_list= [0.62*0.262* 1.292 * 0.5, 
                     0.62*0.262* 1.292 * 0.5, 
                     0.34*0.01* 1.292 * 0.5, 
                     0.34*0.1* 1.292 * 0.5, 
                     1.08*0.31* 1.292 * 0.5]

        self.cp_list=[np.array([-0.0,0.5,0.0],       dtype=np.float).flatten(), \
                    np.array([-0.0,-0.5,0.0],      dtype=np.float).flatten(), \
                    np.array([-1.00,0.85,0],    dtype=np.float).flatten(),\
                    np.array([-1.00,-0.85,0],   dtype=np.float).flatten(),\
                    np.array([0.0,0,0.0],          dtype=np.float).flatten()]

        theta=45.0/180.0*np.pi
        
        Rvd=np.array([[1.0,0.0,0.0],
                        [0.0,np.cos(theta),np.sin(theta)],
                        [0.0,-np.sin(theta),np.cos(theta)]])
        
        Rvg=np.array([[1.0,0.0,0.0],
                        [0.0,np.cos(theta),-np.sin(theta)],
                        [0.0,np.sin(theta),np.cos(theta)]])
        
        
        self.forwards=[np.array([1.0,0,0])]*2
        self.forwards.append(Rvd@np.array([1.0,0,0]))
        self.forwards.append(Rvg@np.array([1.0,0,0]))
        self.forwards.append(np.array([1.0,0,0]))
            
        self.upwards=[np.array([0.0,0,1.0])]*2
        self.upwards.append(Rvd@np.array([0.0,0,1.0]))
        self.upwards.append(Rvg@np.array([0.0,0,1.0]))
        self.upwards.append(np.array([0.0,0,1.0]))
            
        self.crosswards=[np.cross(j,i) for i,j in zip(self.forwards,self.upwards)]

        self.moy_rotor_speed = self.Dict_variables["rotor_moy_speed"]

            
        # self.Dict_variables = OrderedDict(sorted(self.Dict_variables.items(), key=lambda t: t[0]))
            
       
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


        " called from opti refers to wether this class is instanciated in a simulation"
        "or in a optimization process, in which case called_from_opti==True "
        "This blocks the logging process, so that calling the MoteurPhysique in an optimization"
         "does not create a log as a simulation would (would be confusing)"


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
                mp.write(open("MoteurPhysique_class_bis.py").read())

            print(self.save_path_base)
    
    def orthonormalize(self,R_i):
        R=R_i
        R[:,0]/=np.linalg.norm(R[:,0])
        R[:,1]=R[:,1]-np.dot(R[:,0].flatten(),R[:,1].reshape((3,1)))*R[:,0]
        R[:,1]/=np.linalg.norm(R[:,1])
        R[:,2]=np.cross(R[:,0],R[:,1])
        R[:,2]/=np.linalg.norm(R[:,2])
        return R



    def Init(self,joystick_input,t):

        "initialize MoteurPhysique"

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
                                                                     
        return 
    
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
    


    def compute_dynamics(self,joystick_input,t):


        " the flight dynamics are hard coded in this method"


        self.Init(joystick_input,t)
            
        if t<self.T_init:
            self.forces= np.array([0,0,0]) 
            self.torque= np.array([0,0,0])
            if not self.called_from_opti:
                return print("Début des commandes dans :", self.T_init-t)
                
        else: 

            "calculations are performed taking advantage of numpy vectorization"

            "project airspeed in liftdrag plane "    

            v_body=self.R.T@self.speed
            v_in_ldp= v_body - np.cross(self.cp_list,self.omega)
        
            v_in_ldp = - np.cross(self.crosswards,np.cross(self.crosswards,v_in_ldp))
    
            "drag is opposite to airspeed projection in liftdrag plane "    

            dd=-v_in_ldp
            dd=dd.T@np.diag(1.0/(np.linalg.norm(dd,axis=1)+1e-8))
        
            "lift is perpendicular to airspeed, directed upwards in liftdrag plane "    

            ld=np.cross(self.crosswards,v_in_ldp)
            ld=ld.T@np.diag(1.0/(np.linalg.norm(ld,axis=1)+1e-8))    
    


            "the name Aire_list is abusive, actually, its elements are not surfaces but 0.5*surface * density"

            dragdirs=self.R@(dd@np.diag(self.Aire_list)*np.linalg.norm(v_in_ldp)**2)
            liftdirs=self.R@(ld@np.diag(self.Aire_list)*np.linalg.norm(v_in_ldp)**2)
           
            " compute angle of attack as the angle between the frontward vector and airspeed projected in LDP"

            alphas_d=np.diag(v_in_ldp@(np.array(self.forwards).T))/(np.linalg.norm(v_in_ldp,axis=1)+1e-8)
            alphas_d=np.arccos(alphas_d)
            alphas_d=np.sign(np.diag(v_in_ldp@np.array(self.upwards).T))*alphas_d   


            self.Dict_etats['alpha']=alphas_d
        
            "using alpha, and the physical parameters, compute C_L and C_L for each wing"
            a=alphas_d
            a_0_arr=self.Dict_variables['alpha0']
            
            CL_sa = 1/2 * self.Dict_variables['cl1sa'] * np.sin(2*(a  + a_0_arr \
                                         + self.Dict_variables["coeff_lift_shift"]*self.Dict_Commande["delta"] ))
            CD_sa = self.Dict_variables['cd0sa'] +\
                                        self.Dict_variables['cd1sa'] * np.sin(a + a_0_arr \
                                         + self.Dict_variables["coeff_drag_shift"]*self.Dict_Commande["delta"])**2
            
            CL_fp = 1/2 * self.Dict_variables['cl1fp'] * np.sin(2*(a   \
                                         +self.Dict_variables["coeff_lift_shift"]*self.Dict_Commande["delta"] ))
                
            CD_fp = self.Dict_variables['cd0fp'] +\
                                        self.Dict_variables['cd1fp'] * np.sin(a +  \
                                         + self.Dict_variables["coeff_drag_shift"]*self.Dict_Commande["delta"])**2

            a_s,d_s=self.Dict_variables['alpha_stall'],self.Dict_variables['largeur_stall']

            "only thing left to do is compyting sigma to achive, sa/fp model fusion"

            s = np.where(
                abs(a+a_0_arr)>a_s+d_s,
                         
                     np.zeros(len(np.array([a_0_arr]))),
                 
                     np.where(abs(a+a_0_arr)>a_s,
                              
                          0.5*(1+np.cos(np.pi*(a+a_0_arr-np.sign(a+a_0_arr)*a_s)/d_s)),
                          
                          np.ones(len(np.array([a_0_arr]))))
                 )
            
            for i in s:
                if i<1.0:
                    print("STALL")
            
            "below are the final aerodynamical coefficients"

            C_L = CL_fp + s*(CL_sa - CL_fp)   + self.Dict_variables['coeff_lift_gain'] * np.sin(self.Dict_Commande["delta"])
            C_D = CD_fp + s*(CD_sa - CD_fp) 
            
            lifts=C_L*liftdirs    
            drags=C_D*dragdirs        

            aeroforce_total=np.sum(lifts+drags,axis=1)
            
            T=self.R[:,0]*self.Dict_Commande["rotor_speed"]**2*self.Dict_variables['Ct']
            g=np.array([0,0,9.81])
            m=8.5
            self.forces= T + m*g +  aeroforce_total

            self.torque =  np.sum(np.cross(self.cp_list,
                                           np.transpose(self.R.T@(lifts+drags))),axis=0)
            
            if self.takeoff==0:
                self.forces[2]=min(self.forces[2],0)
                
            return

    
    
    def update_state(self,dt):
        "wrapper for compute dynamics"

        "this allows handling specific cases, e.g prevent freefall when the simulator boots "
        "and there is no joystick inputs, etc.... "

        "the euler scheme is implemented here"

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

        "update forces"
                
        self.acc=self.forces/m
        self.speed=self.speed+self.acc*dt
        self.pos=self.pos+self.speed*dt        

    def log_state(self):
        
        "logging function, nothing special to say... "

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

        if 'log.txt' not in os.listdir(self.save_path_base):
            print("Here: Init")
            first_line=""
            for j,key in enumerate(keys):
                if j!=0:
                    first_line=first_line+","
                first_line=first_line+key
                
            first_line=first_line+"\n"
            with open(os.path.join(self.save_path_base,"log.txt"),'a') as f:
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
        
        with open(os.path.join(self.save_path_base,"log.txt"),'a+') as f:
            #print("Here: data")
            f.write(line_to_write)        
            

    def update_sim(self,t,joystick_input):
        
        dt=t-self.last_t
        self.last_t=t
        self.compute_dynamics(joystick_input,self.last_t)
        self.update_state(dt)
        self.log_state()
        
        return
    

