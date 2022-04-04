
import numpy as np
import transforms3d as tf3d
import scipy
from multiprocessing import Pool
import matplotlib.pyplot as plt
import os
import scipy.optimize
import pandas as pd


# %%   ####### IMPORT DATA 

"this script can handle both simulation and real datasets"
"the real dataset is the intellectual property of Sorbonne Universite and thus cannot be disclosed online"
"feel free to use this script to fit the model on sim logs however. HF"

list_of_datasets_name = ["real","sa","sa_noisy","fp","fp_noisy"]
list_of_modesl = ['neural','simple','full']

def gen_grad_acc(df):

    "compute acc in body frame with  numerical derivation of the speedd"
    "can be handy for real datasets, where the raw acceleration comes from the noisy accelerometer"
    "in which case finite differences on the estimated speed is smoother "


    
    df3=df
    
    acc_ned_grad=np.zeros((len(df3),3))
    acc_ned_grad[:,0]=np.diff(df3['speed[0]'],append=0)/np.diff(df3["t"],append=4e-3)
    acc_ned_grad[:,0]=np.array([i  if abs(i)<30 else 0 for i in acc_ned_grad[:,0]])
    acc_ned_grad[:,1]=np.diff(df3['speed[1]'],append=0)/np.diff(df3["t"],append=4e-3)
    acc_ned_grad[:,1]=np.array([i  if abs(i)<30 else 0 for i in acc_ned_grad[:,1]])
    acc_ned_grad[:,2]=np.diff(df3['speed[2]'],append=0)/np.diff(df3["t"],append=4e-3)
    acc_ned_grad[:,2]=np.array([i  if abs(i)<30 else 0 for i in acc_ned_grad[:,2]])
    
    acc_body_grad=np.zeros((len(df3),3))

    for i in df3.index:
        q0,q1,q2,q3=df3["q[0]"][i],df3["q[1]"][i],df3["q[2]"][i],df3["q[3]"][i]
        R=tf3d.quaternions.quat2mat(np.array([q0,q1,q2,q3]))
        acc_body_grad[i]=R.T@(acc_ned_grad[i].reshape((3,1))).flatten()
        
    R_array=np.array([tf3d.quaternions.quat2mat([i,j,k,l]) for i,j,k,l in zip(df['q[0]'],df['q[1]'],df['q[2]'],df['q[3]'])])
    v_ned_array=np.array([df['speed[%i]'%(i)] for i in range(3)]).T
    v_body_array=np.array([(i.T@(j.T)).T for i,j in zip(R_array,v_ned_array)])
    gamma_array=np.array([(i.T@(np.array([0,0,9.81]).T)).T for i in R_array])

    df3['acc_ned_grad[0]'],df3['acc_ned_grad[1]'],df3['acc_ned_grad[2]']=acc_ned_grad.T
    df3['acc_body_grad[0]'],df3['acc_body_grad[1]'],df3['acc_body_grad[2]']=acc_body_grad.T
    
    for i in range(3):
        df3['speed_body[%i]'%(i)]=v_body_array[:,i]
        df3['gamma[%i]'%(i)]=gamma_array[:,i]
    return df3



for ds in list_of_datasets_name:
    for model in ["neural"]:
        

        " import and prepare data "
        log_path=os.path.join(os.getcwd(),"datasets",ds,"log.txt")
        save_dir=os.path.join(os.getcwd(),"datasets",ds,"opti_res")
                              
        os.makedirs(save_dir,exist_ok=True)
            
        raw_data=pd.read_csv(log_path)
        
        
        
        prep_data=raw_data.drop(columns=[i for i in raw_data.keys() if (("forces" in i ) or ('pos' in i) ) ])
        prep_data=prep_data[100:-10]
        
            
        prep_data=prep_data.drop(columns=[i for i in raw_data.keys() if (("level" in i ) or ('Unnamed' in i) or ("index" in i)) ])
        prep_data=prep_data.reset_index()


        if ds=="real":
            prep_data["takeoff"]=np.ones(len(prep_data))
        
        
        for i in range(3):
            prep_data['speed_pred[%i]'%(i)]=np.r_[prep_data['speed[%i]'%(i)].values[1:len(prep_data)],0]
            
            
        prep_data['dt']=np.r_[prep_data['t'].values[1:]-prep_data['t'].values[:-1],0]
        prep_data['t']-=prep_data['t'][0]
        prep_data=prep_data.drop(index=[0,len(prep_data)-1])
        prep_data=prep_data.reset_index()
        
        df=prep_data[:len(prep_data)]

        "this was specific to toy datasets produced for demonstration"
        " basically, you can specify here where to cut, typically you should cutoff "
        " takeoff and the last seconds (when you drop the joystick to close the window)"

        if "sa" in ds:
            df = df[df['t']<85]
        elif "fp" in ds:
            df = df[df['t']<80]

        
        df = gen_grad_acc(df).drop(columns=['level_0','index'])


        if "real" not in ds:  

            "we just use the same parameters as in the moteur physique"
            "remember to change it here if you change it in moteurphysique class"
            "however changing working values is not a good idea tbh."



            CTSIM=2*2e-4
            Tint = CTSIM*df['rot_speed']**2
            df.insert(df.shape[1],'thrust_intensity',Tint)
            delt=np.array([df['delta[%i]'%(i)] for i in range(4)]).T
            delt=np.concatenate((delt,np.zeros((len(delt),1))),axis=1)
            delt=delt.reshape((delt.shape[0],1,delt.shape[1]))

            "to compute and log acc,alpha,q,omega at step k : "
            " rot_speed k"
            " deltas_k "
            " q k-1"
            " omega k-1"
            " speed k-1 "
            
            df.insert(df.shape[1],
                      'deltas',
                      [i for i in delt])
            
            keys_kp1=["acc","alpha","rot_speed","delta",'thrust_','takeoff']
            for i in df.keys():
                for k in keys_kp1:
                    if k in i:
                        df[i][:-1]=df[i][1:] 
                        if i=="takeoff":
                            df[i][:-1]=df[i][1:] 
                            "le takeoff est décalé de 2dt"
            df=df[:-5]
            "splitting the dataset into nsecs sec minibatches"    

        else:

            "this is specific to the azimut dataset"

            df.insert(df.shape[1],'omega_c[5]',df['PWM_motor[5]']*9.57296629e-01-1.00188650e+03)
            radsec_reg = df['PWM_motor[5]']*9.57296629e-01-1.00188650e+03
            df['rot_speed']=radsec_reg

            Tint =np.clip(5.74144050e-05*radsec_reg**2 -6.49277834e-03*radsec_reg -1.15899198e+00,0,1e10)
            # Tint =np.clip(1.5e-04*radsec_reg**2 ,0,1e10)
        
            df.insert(df.shape[1],'thrust_intensity',Tint)

            delt=np.array([df['PWM_motor[%i]'%(i)] for i in range(1,5)]).T
            delt=np.concatenate((np.zeros((len(df),1)),delt),axis=1).reshape(-1,1,5)
            delt=(delt-1530)/500*30.0/180.0*np.pi 
            delt[:,:,0]*=0
            delt[:,:,2]*=-1.0
            delt[:,:,4]*=-1.0
            delt=delt-np.mean(delt,axis=0)
            
            df.insert(df.shape[1],
                      'deltas',
                      [i for i in delt])


        df.insert(df.shape[1],
                  'R',
                  [tf3d.quaternions.quat2mat([i,j,k,l]) for i,j,k,l in zip(df['q[0]'],df['q[1]'],df['q[2]'],df['q[3]'])])
        
        df.insert(df.shape[1],
                              'Thrust_Reg',
                              [i[:,0]*j for i,j in zip(df['R'],df['thrust_intensity'])])
        
        R_array=np.array([i for i in df["R"]])
        
        v_ned_array=np.array([df['speed[%i]'%(i)] for i in range(3)]).T
        v_body_array=np.array([(i.T@(j.T)).T for i,j in zip(R_array,v_ned_array)])
        gamma_array=np.array([(i.T@(np.array([0,0,9.81]).T)).T for i in R_array])
        
        for i in range(3):
            try:
                df.insert(df.shape[1],
                          'speed_body[%i]'%(i),
                          v_body_array[:,i])
            except:
                pass
        



        if "real" not in ds:

            "reimplementing the same equations and parameters as in simulator class"

            Aire_list = [0.62*0.262* 1.292 * 0.5, 
                         0.62*0.262* 1.292 * 0.5, 
                         0.34*0.01* 1.292 * 0.5, 
                         0.34*0.1* 1.292 * 0.5, 
                         1.08*0.31* 1.292 * 0.5]
            
            cp_list=[np.array([-0.0,0.5,0.0],       dtype=np.float).flatten(), \
                        np.array([-0.0,-0.5,0.0],      dtype=np.float).flatten(), \
                        np.array([-1.00,0.85,0],    dtype=np.float).flatten(),\
                        np.array([-1.00,-0.85,0],   dtype=np.float).flatten(),\
                        np.array([0.0,0,0.0],          dtype=np.float).flatten()]
            #1 : aile droite
            #2 : aile gauche
            #3 : vtail droit 
            #4 : vtail gauche
            #5 : aile centrale
            
            theta=45.0/180.0*np.pi
            
            Rvd=np.array([[1.0,0.0,0.0],
                            [0.0,np.cos(theta),np.sin(theta)],
                            [0.0,-np.sin(theta),np.cos(theta)]])
            
            Rvg=np.array([[1.0,0.0,0.0],
                            [0.0,np.cos(theta),-np.sin(theta)],
                            [0.0,np.sin(theta),np.cos(theta)]])
            
            forwards=[np.array([1.0,0,0])]*2
            forwards.append(Rvd@np.array([1.0,0,0]))
            forwards.append(Rvg@np.array([1.0,0,0]))
            forwards.append(np.array([1.0,0,0]))
                
            upwards=[np.array([0.0,0,1.0])]*2
            upwards.append(Rvd@np.array([0.0,0,1.0]))
            upwards.append(Rvg@np.array([0.0,0,1.0]))
            upwards.append(np.array([0.0,0,1.0]))
                
            crosswards=[np.cross(j,i) for i,j in zip(forwards,upwards)]

            omegas=np.c_[df['omega[0]'],df['omega[1]'],df['omega[2]']]
            
        else:
   
            
            Aire_1,Aire_2,Aire_3,Aire_4,Aire_0 =    0.62*0.262* 1.292 * 0.5,\
                                                0.62*0.262* 1.292 * 0.5, \
                                                0.34*0.1* 1.292 * 0.5,\
                                                0.34*0.1* 1.292 * 0.5, \
                                                1.08*0.31* 1.292 * 0.5
                                                
            Aire_list = [Aire_0,Aire_1,Aire_2,Aire_3,Aire_4]
            
            cp_1,cp_2,cp_3,cp_4,cp_0 = np.array([-0.013,0.475,-0.040],       dtype=float).flatten(), \
                                    np.array([-0.013,-0.475,-0.040],      dtype=float).flatten(), \
                                    np.array([-1.006,0.17,-0.134],    dtype=float).flatten(),\
                                    np.array([-1.006,-0.17,-0.134],   dtype=float).flatten(),\
                                    np.array([0.021,0,-0.064],          dtype=float).flatten()
            cp_list=[cp_0,cp_1,cp_2,cp_3,cp_4]
            
            #0 : aile centrale
            #1 : aile droite
            #2 : aile gauche
            #3 : vtail droit 
            #4 : vtail gauche
            
            theta=45.0/180.0/np.pi
            
            Rvd=np.array([[1.0,0.0,0.0],
                            [0.0,np.cos(theta),np.sin(theta)],
                            [0.0,-np.sin(theta),np.cos(theta)]])
            
            Rvg=np.array([[1.0,0.0,0.0],
                            [0.0,np.cos(theta),-np.sin(theta)],
                            [0.0,np.sin(theta),np.cos(theta)]])
            
            
            forwards=[np.array([1.0,0,0])]*3
            forwards.append(Rvd@np.array([1.0,0,0]))
            forwards.append(Rvg@np.array([1.0,0,0]))
            
            upwards=[np.array([0.0,0,1.0])]*3
            upwards.append(Rvd@np.array([0.0,0,1.0]))
            upwards.append(Rvg@np.array([0.0,0,1.0]))
            
            crosswards=[np.cross(j,i) for i,j in zip(forwards,upwards)]

            "in the case of the real dataset, omega can be recomputed using finite differences"
            " on the attitude "

            def skew_to_x(S):
                SS=(S-S.T)/2
                return np.array([SS[1,0],SS[2,0],S[2,1]])
            
            def skew(x):
                return np.array([[0,-x[2],x[1]],
                                  [x[2],0,-x[0]],
                                  [-x[1],x[0],0]])
            
            omegas=np.zeros((R_array.shape[0],3))
            omegas[1:]=[skew_to_x(j@(i.T)-np.eye(3)) for i,j in zip(R_array[:-1],R_array[1:])]
            omegas[:,0]=omegas[:,0]*1.0/df['dt']
            omegas[:,1]=omegas[:,1]*1.0/df['dt']
            omegas[:,2]=omegas[:,2]*1.0/df['dt']
            
            def filtering(X,k=0.05):
                Xnew=[X[0]]
                for i,x in enumerate(X[1:]):
                    xold=Xnew[-1]
                    xnew=xold+k*(x-xold)
                    Xnew.append(xnew)
                return np.array(Xnew)
            
            omegas=filtering(omegas)
        

        "now we compute the dynamics "
        "again we use the numpy vectorization to shorten calculation time"
        "this is precomputed, as this does not involve any identification parameter"

        "the idea is that all the quantities that can be computed"
        " once and for all before the identification procedure should be."

        dragdirs=np.zeros((v_body_array.shape[0],3,5))
        liftdirs=np.zeros((v_body_array.shape[0],3,5))
        slipdirs=np.zeros((v_body_array.shape[0],3,5))
        
        alphas=np.zeros((v_body_array.shape[0],1,5))
        sideslips=np.zeros((v_body_array.shape[0],1,5))
        
        for k,v_body in enumerate(v_body_array):
            
            v_in_ldp= v_body - np.cross(cp_list,omegas[k])
            v_in_ldp = - np.cross(crosswards,np.cross(crosswards,v_in_ldp))
        
            dd=-v_in_ldp
            dd=dd.T@np.diag(1.0/(np.linalg.norm(dd,axis=1)+1e-8))
        
            ld=np.cross(crosswards,v_in_ldp)
            ld=ld.T@np.diag(1.0/(np.linalg.norm(ld,axis=1)+1e-8))    
        
            dragdirs[k,:,:]=R_array[k]@(dd@np.diag(Aire_list)*np.linalg.norm(v_in_ldp)**2)
            liftdirs[k,:,:]=R_array[k]@(ld@np.diag(Aire_list)*np.linalg.norm(v_in_ldp)**2)
           
            alphas_d=np.diag(v_in_ldp@(np.array(forwards).T))/(np.linalg.norm(v_in_ldp,axis=1)+1e-8)
            alphas_d=np.arccos(alphas_d)
            alphas_d=np.sign(np.diag(v_in_ldp@np.array(upwards).T))*alphas_d   
            alphas[k,:,:]=alphas_d
            
        df.insert(df.shape[1],
                  'liftdirs',
                  [i for i in liftdirs])
                
        df.insert(df.shape[1],
                  'dragdirs',
                  [i for i in dragdirs])     
        
        df.insert(df.shape[1],
                  'alphas_calc',
                  [i for i in alphas])    
        
        df.insert(df.shape[1],
                  'sideslips',
                  [i for i in sideslips])    
        
        # %% OPTI INTERM
        

        "this is used to generate a random parammeter init set, when working with simulated datasets "
        "where the 'real' values of the parameter are exactly known"

        def generate_random_params(X_params,amp_dev=0.0):
            return X_params*(1+amp_dev*(np.random.random(size=len(X_params))-0.5))   
        
        
        "hereby are defined the identification procedure for three models"
        " simple where only the small angles model is implemented"
        " full where both small angles and fp are implemented "
        " and neural."

        "if the dataset contains only data generated at small angles, both simple and full models converge well"
        " however, if the dataset contains data generated with high angles, the simple model won't fit perfectly (obviously)"
        " but the full model will still converge"

        "neural is here to compete with the full and simple models that we known will perfectly generated data"



        if model=="simple":
            
            a_0_real=0.0
            cl1sa_real=5.0
            lift_gain_real=2.5
            cd0sa_real=0.025
            cd1sa_real=4.55
            m_real=8.5
            ct_real=2*2e-4
            
            
            bounds_ct= (0,1) #1.1e-4/4 1.1e-4*4
            bounds_ct10 = (0,1)
            bounds_ct20=(None,None)
            bounds_a_0= (-0.20,0.20) #7 deg
            bounds_a_s= (-np.pi/4,np.pi/4)  #2 30 deg
            bounds_d_s= (-np.pi/4,np.pi/4) #2 45 degres
            bounds_cl1sa =(0,None)
            bounds_cl1fp =(0,None)
            bounds_cd1fp =(0,None)
            bounds_k0 =(None,None)
            bounds_k1 =(None,None)
            bounds_lift_gain =(None,None)
            bounds_cd0fp =(0,None)
            bounds_cd0sa =(0,None)
            bounds_cd1sa= (0,None)
            bounds_mass=(5,15)
            
            coeffs_interm_0=generate_random_params(np.array([ct_real,
                             a_0_real, 
                           cl1sa_real,
                           lift_gain_real,
                           cd0sa_real,
                           cd1sa_real,
                           m_real]),1.0)
            
            bounds_interm=[bounds_ct,bounds_a_0,bounds_cl1sa,
                          bounds_lift_gain,bounds_cd0sa,
                          bounds_cd1sa,bounds_mass]
            
            drag_dirs_=  np.array([i for i in df["dragdirs"]])
            lift_dirs_=  np.array([i for i in df["liftdirs"]])
            
            def dyn_interm(df=df,coeffs=coeffs_interm_0,fc=False,fm=False):
                
                ct,a_0, cl1sa, lift_gain ,cd0sa, cd1sa,m=coeffs
                m = m_real if fm else m
                ct= ct_real if fc else ct
                
                a = np.array([i for i in df['alphas_calc']])
                d_0 = np.array([i for i in df['deltas']])
                a_0_arr = a_0*np.ones(a.shape)
                # a_0_arr[:,:,2:4]*=0
                
                CL_sa = 1/2 * cl1sa * np.sin(2*(a  + a_0_arr  ))
                CD_sa = cd0sa + cd1sa * np.sin((a + a_0_arr ))**2
            
                C_L = CL_sa  + lift_gain * np.sin(d_0)
                C_D = CD_sa 
            
                ld,dd=np.array([i for i in df['liftdirs']]),np.array([i for i in df['dragdirs']])
            
                lifts=C_L*ld    
                drags=C_D*dd
            
                aeroforce_total=np.sum(lifts+drags,axis=2)
            
                T=np.array([i for i in df['Thrust_Reg']]) if fc else np.array([i[:,0]*j for i,j in zip(R_array,ct*df['rot_speed']**2)])
            
                g=np.zeros(aeroforce_total.shape)
                g[:,-1]+=9.81
            
                forces_total=T+aeroforce_total+m*g
                acc=forces_total/m
                acc[df["takeoff"]==0,2]*=0
                return acc
            
            if "real" in ds:
                acc_log=np.array([df['acc_ned_grad[%i]'%(i)] for i in range(3)]).T 
            else:
                acc_log=np.array([df['acc[%i]'%(i)] for i in range(3)]).T 
            pd.DataFrame(data=np.c_[df['t'].values,acc_log],columns=["t"]+["acc_log[%i]"%(i) \
                   for i in range(3)]).to_csv(os.path.join(save_dir,"acc_log_%s.csv"%(ds)))
                
            def cost_interm(X,fm=False,fc=False,
                            scaling=True):
                
                X0=X*coeffs_interm_0 if scaling else X
                
                acc=dyn_interm(df,X0,fc=fc,fm=fm)   
                # c=np.mean(np.linalg.norm((acc-acc_log),axis=1))
                c=np.mean(np.linalg.norm((acc-acc_log),axis=1)/np.maximum(0.1,np.linalg.norm(acc_log,axis=1)))
                str_top_print="\r "
                for i in X:
                    str_top_print=str_top_print+str(round(i,ndigits=5))+" |"
                str_top_print=str_top_print+" "+str(round(c,ndigits=5))
                
                res={}
                l="ct,a_0,cl1sa,lift_gain,cd0sa,cd1sa,m"
                for i,j in zip(l.split(","),X0):
                    res[i]=round(j,ndigits=5)       
                res['cost']=c
                print(res)
                return c
            
            def run_parallel_interm(x):
                    fm,fc,scaling,bnds,deviation_param=x
                    
                    x0=np.ones(len(coeffs_interm_0)) if scaling else coeffs_interm_0
                    x0=generate_random_params(x0,amp_dev=deviation_param)
                    
                    if bnds:
                        sol=scipy.optimize.minimize(cost_interm,x0,
                        args=(fm,fc,scaling),bounds=bounds_interm)
                    else:
                        sol=scipy.optimize.minimize(cost_interm,x0,
                        args=(fm,fc,scaling))
                        
                    filename="INTERM_fm_"+str(fm)
                    filename=filename+"_fc_"+str(fc)
                    filename=filename+"_scaling_"+str(scaling)
                    filename=filename+"_bounds_"+str(bnds)
                    filename=filename+"_dev_"+str(deviation_param)
                    
                    sfile=os.path.join(save_dir,'%s.csv'%(filename))
                    keys='cost,fm,fc,scaling,bnds,dev,ct,a_0,cl1sa,lift_gain,cd0sa,cd1sa,m'.replace(' ','').split(",")
            
                    data=[sol['fun']]
                    data=data+list(x)
                    data=data+(coeffs_interm_0*sol['x']).tolist() if scaling else data+sol['x'].tolist()
            
                    df_save=pd.DataFrame(data=[data],columns=keys)
                    df_save.to_csv(sfile)
                    
                    pred_coeffs=coeffs_interm_0*sol['x'] if scaling else sol['x']
                    
                    acc_pred=dyn_interm(df=df,coeffs=pred_coeffs,
                                      fm=True,fc=True)
                    
                    filename="acc_pred_INTERM_fm_"+str(fm)
                    filename=filename+"_fc_"+str(fc)
                    filename=filename+"_scaling_"+str(scaling)
                    filename=filename+"_bounds_"+str(bnds)
                    filename=filename+"_dev_"+str(deviation_param)
                    sfile=os.path.join(save_dir,filename+".csv")
                    acc_pred_df=pd.DataFrame(data=acc_pred,columns=['acc_pred[0]',
                                                                      'acc_pred[1]',
                                                                      'acc_pred[2]'])
                    
                    acc_pred_df.to_csv(sfile)              
                    
                    return
                
            if __name__ == '__main__':
            
                fm_range=[True,False]
                fc_range=[True,False]
                sc_range=[True,False]
                bnds_range=[True,False]
            
                x_r=[]
                for j in  fm_range :
                    for k in fc_range:
                        for l in sc_range:
                                for p in bnds_range:
                                    for d in [0.0]:
                                        x_r.append([j,k,l,p,d])
            
                pool = Pool(processes=16)
                pool.map(run_parallel_interm, x_r)
                
        
        if model=="full":
            
            a_0_real=0.0
            cl1sa_real=5.0
            lift_gain_real=2.5
            cd0sa_real=0.025
            cd1sa_real=4.55
            m_real=8.5
            ct_real=2*2e-4
            
            a_s_real=15.0*np.pi/180
            d_s_real=15.0*np.pi/180
            cl1fp_real=3.65
            lift_shift_real=0
            drag_shift_real=0
            cd0fp_real=0.025
            cd1fp_real=2.0

            bounds_ct= (0,1) #1.1e-4/4 1.1e-4*4
            bounds_ct10 = (0,1)
            bounds_ct20=(None,None)
            bounds_a_0= (-0.20,0.20) #7 deg
            bounds_a_s= (-np.pi/4,np.pi/4)  #2 30 deg
            bounds_d_s= (0,np.pi/4) #2 45 degres
            bounds_cl1sa =(0,None)
            bounds_cl1fp =(0,None)
            bounds_cd1fp =(0,None)
            bounds_k0 =(None,None)
            bounds_k1 =(None,None)
            bounds_lift_gain =(None,None)
            bounds_cd0fp =(0,None)
            bounds_cd0sa =(0,None)
            bounds_cd1sa= (0,None)
            bounds_mass=(5,15)

            coeffs_full_0=generate_random_params(np.array([ct_real, 
                        a_0_real, 
                        a_s_real, 
                        d_s_real, 
                        cl1sa_real, 
                        cl1fp_real, 
                        lift_gain_real, 
                        lift_shift_real, 
                        drag_shift_real, 
                        cd0fp_real, 
                        cd0sa_real, 
                        cd1sa_real, 
                        cd1fp_real, 
                        m_real]),1.0)
            
            bounds_full=[bounds_ct,bounds_a_0,
                         bounds_a_s,bounds_d_s,
                         bounds_cl1sa,bounds_cl1fp,
                         bounds_lift_gain,bounds_lift_gain,bounds_lift_gain,
                         bounds_cd0sa,bounds_cd0sa,bounds_cd0sa,bounds_cd0sa,
                         bounds_mass]
            
            def dyn_full(df=df,coeffs=coeffs_full_0,fc=False,fm=False):
                
                ct, a_0, a_s, d_s, cl1sa, cl1fp, lift_gain,lift_shift,drag_shift\
                    ,cd0fp, cd0sa, cd1sa, cd1fp,m = coeffs
            
                m = m_real if fm else m
                ct= ct_real if fc else ct
                
                a = np.array([i for i in df['alphas_calc']])
                d_0 = np.array([i for i in df['deltas']])
            
                a_0_arr = a_0*np.ones(a.shape)
                
                CL_sa = 1/2 * cl1sa * np.sin(2*(a  + a_0_arr  ))
                CD_sa = cd0sa + cd1sa * np.sin((a + a_0_arr ))**2
            
                CL_fp = 1/2 * cl1fp * np.sin(2*(a   \
                                                +lift_shift*d_0 ))
                    
                CD_fp = cd0fp + cd1fp * np.sin(a +  \
                                                + drag_shift*d_0)**2
               
                s = np.where(
                    abs(a+a_0_arr)>a_s+d_s,
                             
                          np.zeros(len(np.array([a_0_arr]))),
                     
                          np.where(abs(a+a_0_arr)>a_s,
                                  
                              0.5*(1+np.cos(np.pi*(a+a_0_arr-np.sign(a+a_0_arr)*a_s)/d_s)),
                              
                              np.ones(len(np.array([a_0_arr]))))
                      )
                
                C_L = CL_fp + s*(CL_sa - CL_fp)   + lift_gain * np.sin(d_0)
                C_D = CD_fp + s*(CD_sa - CD_fp) 
            
                ld,dd=np.array([i for i in df['liftdirs']]),np.array([i for i in df['dragdirs']])
            
                lifts=C_L*ld    
                drags=C_D*dd
            
                aeroforce_total=np.sum(lifts+drags,axis=2)
            
                T=np.array([i for i in df['Thrust_Reg']]) if fc else np.array([i[:,0]*j for i,j in zip(R_array,ct*df['rot_speed']**2)])
            
                g=np.zeros(aeroforce_total.shape)
                g[:,-1]+=9.81
            
                forces_total=T+aeroforce_total+m*g
                forces_total[df["takeoff"]==0,2]=np.minimum(forces_total[df["takeoff"]==0][:,2],0)
                acc=forces_total/m
                return acc
            
            if "real" in ds:
                acc_log=np.array([df['acc_ned_grad[%i]'%(i)] for i in range(3)]).T 
            else:
                acc_log=np.array([df['acc[%i]'%(i)] for i in range(3)]).T 
                
            pd.DataFrame(data=np.c_[df['t'].values,acc_log],columns=["t"]+["acc_log[%i]"%(i) \
                   for i in range(3)]).to_csv(os.path.join(save_dir,"acc_log_%s.csv"%(ds)))   
                
            def cost_full(X,fm=False,fc=False,
                            scaling=True):
                
                X0=X*coeffs_full_0 if scaling else X
                
                acc=dyn_full(df,X0,fc=fc,fm=fm)   
                # c=np.mean(np.linalg.norm((acc-acc_log),axis=1))
                c=np.mean(np.linalg.norm((acc-acc_log),axis=1)/np.maximum(0.1,np.linalg.norm(acc_log,axis=1)))

                str_top_print="\r "
                for i in X:
                    str_top_print=str_top_print+str(round(i,ndigits=5))+" |"
                str_top_print=str_top_print+" "+str(round(c,ndigits=5))
                
                res={}
                l="ct, a_0, a_s, d_s, cl1sa, cl1fp, lift_gain,lift_shift,drag_shift\
                    ,cd0fp, cd0sa, cd1sa, cd1fp,m"
                for i,j in zip(l.split(","),X0):
                    res[i]=round(j,ndigits=5)       
                res['cost']=c
                print(res)
                return c
            

            
            def run_parallel_full(x):
                    fm,fc,scaling,bnds,deviation_param=x
                    
                    x0=np.ones(len(coeffs_full_0)) if scaling else coeffs_full_0
                    x0=generate_random_params(x0,amp_dev=deviation_param)
                    
                    if bnds:
                        sol=scipy.optimize.minimize(cost_full,x0,
                        args=(fm,fc,scaling),bounds=bounds_full)
                    else:
                        sol=scipy.optimize.minimize(cost_full,x0,
                        args=(fm,fc,scaling))
                        
                    filename="res_FULL_fm_"+str(fm)
                    filename=filename+"_fc_"+str(fc)
                    filename=filename+"_scaling_"+str(scaling)
                    filename=filename+"_bounds_"+str(bnds)
                    filename=filename+"_dev_"+str(deviation_param)
                    
                    sfile=os.path.join(save_dir,'%s.csv'%(filename))
                    keys='cost,fm,fc,scaling,bnds,dev,ct, a_0, a_s, d_s, cl1sa, cl1fp, lift_gain,lift_shift,drag_shift\
                    ,cd0fp, cd0sa, cd1sa, cd1fp,m'.replace(' ','').split(",")
            
                    data=[sol['fun']]
                    data=data+list(x)
                    data=data+(coeffs_full_0*sol['x']).tolist() if scaling else data+sol['x'].tolist()
            
                    df_save=pd.DataFrame(data=[data],columns=keys)
                    df_save.to_csv(sfile)
                    
                    pred_coeffs=coeffs_full_0*sol['x'] if scaling else sol['x']
                    
                    acc_pred=dyn_full(df=df,coeffs=pred_coeffs,
                                      fm=True,fc=True)
                    
                    filename="acc_pred_FULL_fm_"+str(fm)
                    filename=filename+"_fc_"+str(fc)
                    filename=filename+"_scaling_"+str(scaling)
                    filename=filename+"_bounds_"+str(bnds)
                    filename=filename+"_dev_"+str(deviation_param)
                    sfile=os.path.join(save_dir,filename+".csv")
                    
                    acc_pred_df=pd.DataFrame(data=acc_pred,columns=['acc_pred[0]',
                                                                      'acc_pred[1]',
                                                                      'acc_pred[2]'])
                    
                    acc_pred_df.to_csv(sfile)
                    
                    return
                
            if __name__ == '__main__':
            
                fm_range=[True,False]
                fc_range=[True,False]
                sc_range=[True,False]
                bnds_range=[True,False]
            
                x_r=[]
                for j in  fm_range :
                    for k in fc_range:
                        for l in sc_range:
                                for p in bnds_range:
                                    for d in [0.0]:
                                        x_r.append([j,k,l,p,d])
            
                pool = Pool(processes=16)
                pool.map(run_parallel_full, x_r)
                

        if model=="neural":
            
            
            import tensorflow as tf
            from tensorflow import keras 
            from sklearn.model_selection import train_test_split
            from sklearn.model_selection import KFold
            
                        

            
             
            if "real" in ds:
                X=df[['speed[0]',
                       'speed[1]', 'speed[2]', 'q[0]', 'q[1]', 'q[2]', 'q[3]',
                       'PWM_motor[1]','PWM_motor[2]', 'PWM_motor[3]',
                       'PWM_motor[4]', 'PWM_motor[5]','PWM_motor[6]',"rot_speed","takeoff"]]
                
                for k in X:
                    if "speed" in k:
                        X[k]/=50.0
                    if "delta" in k:
                        X[k]/=np.pi/2
                    if "PWM" in k:
                        X[k]=(X[k]-1500.0)/1000.0
                        
                y=df[['acc_ned_grad[0]','acc_ned_grad[1]','acc_ned_grad[2]']]/10 

            else:
                X=df[['speed[0]',
                       'speed[1]', 'speed[2]', 'q[0]', 'q[1]', 'q[2]', 'q[3]',
                       "delta[0]","delta[1]","delta[2]","delta[3]","rot_speed","takeoff"]]
                
                for k in X:
                    if "speed" in k:
                        X[k]/=50.0
                    if "delta" in k:
                        X[k]/=np.pi/2
                    if 'rot_' in k:
                        X[k]/=300.0
                y=df[['acc[0]','acc[1]','acc[2]']]/10 
                
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
            
            plane_model=tf.keras.Sequential([keras.layers.Dense(13,activation="relu"),
            keras.layers.Dense(13,activation="relu"),
            keras.layers.Dense(13,activation="tanh"),
            keras.layers.Dense(7,activation="tanh"),
            keras.layers.Dense(3)])
            
            plane_model.compile(loss="mean_squared_error",
                          optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                              metrics=[tf.keras.metrics.MeanSquaredError()])
            
            history = plane_model.fit(X_train, y_train, epochs=50,validation_data=(X_test,y_test)) 
            # print(plane_model.summary())
            acc_pred=plane_model.predict(X)*15.0               
            
            if "real" in ds:
                acc_log=np.array([df['acc_ned_grad[%i]'%(i)] for i in range(3)]).T 
            else:
                acc_log=np.array([df['acc[%i]'%(i)] for i in range(3)]).T 
            pd.DataFrame(data=np.c_[df['t'].values,acc_log],columns=["t"]+["acc_log[%i]"%(i) \
                   for i in range(3)]).to_csv(os.path.join(save_dir,"acc_log_%s.csv"%(ds)))  
            # err=np.mean(np.linalg.norm(acc_pred-acc_log,axis=1))
            err=np.mean(np.linalg.norm((acc_pred-acc_log),axis=1)/np.maximum(0.1,np.linalg.norm(acc_log,axis=1)))

            data=[err,True]
            keys=['cost',"is_rnn"]
            df_save=pd.DataFrame(data=[data],columns=keys)
            sfile=os.path.join(save_dir,'rnn_res.csv')
            df_save.to_csv(sfile)
            
            import json
            history_dict = history.history
            json.dump(history_dict, open(save_dir+"/hist", 'w'))
            
            import shutil 
            dir_name=os.path.join(save_dir,"rnn")
            try:
                shutil.rmtree(dir_name)
            except:
                pass
            os.makedirs(dir_name)
            tf.keras.models.save_model(plane_model,dir_name)
            
            acc_pred=plane_model.predict(X)*15.0               
            
            filename="acc_pred_NEURAL.csv"
            sfile=os.path.join(save_dir,filename)
            acc_pred_df=pd.DataFrame(data=acc_pred,columns=['acc_pred[0]',
                                                              'acc_pred[1]',
                                                              'acc_pred[2]'])
            acc_pred_df.to_csv(sfile,index=False)

            # def plot_learning_curves(ax,hist):
            #     loss,val_loss=history.history["loss"], history.history["val_loss"]
            #     ax.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
            #     ax.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
            #     ax.legend(fontsize=14)
            #     ax.grid(True)
                
            # plt.figure()
            # for i in range(3):
                
            #     ax=plt.gcf().add_subplot(3,2,2*i+1)
            #     # ax.plot(df['t'],df['acc[%i]'%(i)],color="black",label="data")
            #     ax.plot(df['t'],df['acc_ned_grad[%i]'%(i)]
            #             ,color="black",label=r"$a_{%s,data}$"%('ijk'[i]),alpha=0.5)
            
            #     ax.plot(df['t'][np.arange(len(acc_pred))],acc_pred[:,i]
            #             ,color="red",label=r"$a_{%s,pred}$"%('ijk'[i]))
            #     plt.grid(),plt.legend(loc=4),plt.gca().set_ylabel('$m/s^{2}$')
                
            # err=np.mean(np.linalg.norm(acc_pred-acc_log,axis=1))
            # plt.suptitle(r'RMS: ||$a_{data} - a_{pred}$|| : %f $m/s^2$'%(err))
                
            # plt.gca().set_xlabel('t')
            
            # ax=plt.gcf().add_subplot(1,2,2)
            # plot_learning_curves(ax,history)
            # plt.gca().set_ylabel('epoch')
            # plt.gca().set_xlabel('loss')
            
            
            # kfold = KFold(n_splits=8, shuffle=True)
            
            # scores = cross_val_score(plane_model, X, y, cv=5 , verbose =1 )
            # print("scores :", scores)