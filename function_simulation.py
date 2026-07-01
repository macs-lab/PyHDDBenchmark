import numpy as np
from control.matlab import *
import control.matlab as ctm
import control as ct
import math
import sys
import multiprocessing
import pickle
import utils
import plant
import print_ASCII as pa
# prints ASCII art of the system
pa.print_system()

def Function_simulation(Sys_Pc_vcm,Sys_Pc_pzt,Sys_Cd_vcm,Sys_Fm_vcm,Sys_Cd_pzt,Sys_Fm_pzt,Ts,Mr_f,iter):
    # Define sim_result as a dictionary
    sim_result = {
    'time': None,
    'freq': None,
    'yc': None,
    'yc_pzt': None,
    'dp': None,
    'df': None,
    'Fr_yc': None,
    'Fr_yc_pzt': None,
    'Fr_dp': None,
    'Fr_df': None
    }
    
    s = ct.TransferFunction.s

    ## Simulation condition
    Mr_p=20                          # Multi-rate for continuous-time system
    Tsc=Ts/Mr_p                      # Sampling time for continuous-time system
    Tsim=1.1                         # End of simulation time

    ## Controlled object
    Sys_Pcd_vcm=ctm.c2d(ctm.ss(Sys_Pc_vcm),Tsc)
    [A_Pcd_vcm,B_Pcd_vcm,C_Pcd_vcm,_]=ctm.ssdata(Sys_Pcd_vcm)
    Sys_Pcd_pzt=ctm.c2d(ctm.ss(Sys_Pc_pzt),Tsc)
    [A_Pcd_pzt,B_Pcd_pzt,C_Pcd_pzt,_]=ctm.ssdata(Sys_Pcd_pzt)

    ## Feedback Controller & Multi-rate filter
    [A_Cd_vcm,B_Cd_vcm,C_Cd_vcm,D_Cd_vcm]=ctm.ssdata(ctm.ss(Sys_Cd_vcm))
    [A_Fm_vcm,B_Fm_vcm,C_Fm_vcm,D_Fm_vcm]=ctm.ssdata(ctm.ss(Sys_Fm_vcm))
    [A_Cd_pzt,B_Cd_pzt,C_Cd_pzt,D_Cd_pzt]=ctm.ssdata(ctm.ss(Sys_Cd_pzt))
    [A_Fm_pzt,B_Fm_pzt,C_Fm_pzt,D_Fm_pzt]=ctm.ssdata(ctm.ss(Sys_Fm_pzt))

    ## Disturbance signal
    # FAN-Induced Disturbance
    Sys_Dp1_c=0.6/(s**2+2*0.008*2200*2*math.pi*s+(2200*2*math.pi)**2)
    Sys_Dp1=ctm.c2d(ctm.ss(Sys_Dp1_c),Tsc)
    [A_Dp1,B_Dp1,C_Dp1,_]=ctm.ssdata(Sys_Dp1)
    Sys_Dp2_c=0.3/(s**2+2*0.005*2937*2*math.pi*s+(2937*2*math.pi)**2)
    Sys_Dp2=ctm.c2d(ctm.ss(Sys_Dp2_c),Tsc)
    [A_Dp2,B_Dp2,C_Dp2,_]=ctm.ssdata(Sys_Dp2)
    Sys_Dp3_c=1/(s**2+2*0.005*3300*2*math.pi*s+(3300*2*math.pi)**2)
    Sys_Dp3=ctm.c2d(ctm.ss(Sys_Dp3_c),Tsc)
    [A_Dp3,B_Dp3,C_Dp3,_]=ctm.ssdata(Sys_Dp3)
    Sys_Dp4_c=0.5/(s**2+2*0.005*3545*2*math.pi*s+(3545*2*math.pi)**2)
    Sys_Dp4=ctm.c2d(ctm.ss(Sys_Dp4_c),Tsc)
    [A_Dp4,B_Dp4,C_Dp4,_]=ctm.ssdata(Sys_Dp4)
    Sys_Dp5_c=0.3/(s**2+2*0.002*3980*2*math.pi*s+(3980*2*math.pi)**2)
    Sys_Dp5=ctm.c2d(ctm.ss(Sys_Dp5_c),Tsc)
    [A_Dp5,B_Dp5,C_Dp5,_]=ctm.ssdata(Sys_Dp5)
    Sys_Dp6_c=1/(s**2+2*0.01*4220*2*math.pi*s+(4220*2*math.pi)**2)
    Sys_Dp6=ctm.c2d(ctm.ss(Sys_Dp6_c),Tsc)
    [A_Dp6,B_Dp6,C_Dp6,_]=ctm.ssdata(Sys_Dp6)
    Sys_Dp7_c=1/(s**2+2*0.008*4380*2*math.pi*s+(4380*2*math.pi)**2)
    Sys_Dp7=ctm.c2d(ctm.ss(Sys_Dp7_c),Tsc)
    [A_Dp7,B_Dp7,C_Dp7,_]=ctm.ssdata(Sys_Dp7)
    Sys_Dp8_c=0.5/(s**2+2*0.002*5072*2*math.pi*s+(5072*2*math.pi)**2)
    Sys_Dp8=ctm.c2d(ctm.ss(Sys_Dp8_c),Tsc)
    [A_Dp8,B_Dp8,C_Dp8,_]=ctm.ssdata(Sys_Dp8)
    Sys_Dp9_c=2/(s**2+2*0.003*5370*2*math.pi*s+(5370*2*math.pi)**2)
    Sys_Dp9=ctm.c2d(ctm.ss(Sys_Dp9_c),Tsc)
    [A_Dp9,B_Dp9,C_Dp9,_]=ctm.ssdata(Sys_Dp9)
    Sys_Dp10_c=15/(s**2+2*0.04*5850*2*math.pi*s+(5850*2*math.pi)**2)
    Sys_Dp10=ctm.c2d(ctm.ss(Sys_Dp10_c),Tsc)
    [A_Dp10,B_Dp10,C_Dp10,_]=ctm.ssdata(Sys_Dp10)
    Sys_Dp11_c=10/(s**2+2*0.008*6660*2*math.pi*s+(6660*2*math.pi)**2)
    Sys_Dp11=ctm.c2d(ctm.ss(Sys_Dp11_c),Tsc)
    [A_Dp11,B_Dp11,C_Dp11,_]=ctm.ssdata(Sys_Dp11)
    Sys_Dp12_c=1.5/(s**2+2*0.003*7670*2*math.pi*s+(7670*2*math.pi)**2)
    Sys_Dp12=ctm.c2d(ctm.ss(Sys_Dp12_c),Tsc)
    [A_Dp12,B_Dp12,C_Dp12,_]=ctm.ssdata(Sys_Dp12)
    Sys_Dp13_c=2.5/(s**2+2*0.07*9200*2*math.pi*s+(9200*2*math.pi)**2)
    Sys_Dp13=ctm.c2d(ctm.ss(Sys_Dp13_c),Tsc)
    [A_Dp13,B_Dp13,C_Dp13,_]=ctm.ssdata(Sys_Dp13)

    # RRO
    RRO_data= np.loadtxt("Data_RRO.txt", comments="#", delimiter=",", unpack=False)*0.5e-10
    num_sector=420

    # Rotational vibration
    Sys_Df_c=3e-10*(s+50*2*math.pi)/(s+3*2*math.pi)*(s**2+2*20*2000*2*math.pi*s+(2000*2*math.pi)**2)/(s**2+2*0.1*250*2*math.pi*s+(250*2*math.pi)**2)
    Sys_Df=ctm.c2d(ctm.ss(Sys_Df_c),Tsc)
    [A_Df,B_Df,C_Df,_]=ctm.ssdata(Sys_Df)

    # Random signal for disturbance
    np.random.seed(1)  # Set the random seed to 1
    u_random1 = np.random.rand(round(Tsim/Tsc)+1, 1) - 0.5 # generates random numbers between 0 and 1, and subtracting 0.5 shifts the range to -0.5 to 0.5
    np.random.seed(2)
    u_random2 = np.random.rand(round(Tsim/Tsc)+1, 1) - 0.5
    np.random.seed(3)
    u_random3 = np.random.rand(round(Tsim/Tsc)+1, 1) - 0.5
    np.random.seed(4)
    u_random4 = np.random.rand(round(Tsim/Tsc)+1, 1) - 0.5
    np.random.seed(5)
    u_random5 = np.random.rand(round(Tsim/Tsc)+1, 1) - 0.5
    np.random.seed(6)
    u_random6 = np.random.rand(round(Tsim/Tsc)+1, 1) - 0.5
    np.random.seed(7)
    u_random7 = np.random.rand(round(Tsim/Tsc)+1, 1) - 0.5
    np.random.seed(8)
    u_random8 = np.random.rand(round(Tsim/Tsc)+1, 1) - 0.5
    np.random.seed(9)
    u_random9 = np.random.rand(round(Tsim/Tsc)+1, 1) - 0.5
    np.random.seed(10)
    u_random10 = np.random.rand(round(Tsim/Tsc)+1, 1) - 0.5
    np.random.seed(11)
    u_random11 = np.random.rand(round(Tsim/Tsc)+1, 1) - 0.5
    np.random.seed(12)
    u_random12 = np.random.rand(round(Tsim/Tsc)+1, 1) - 0.5
    np.random.seed(13)
    u_random13 = np.random.rand(round(Tsim/Tsc)+1, 1) - 0.5
    np.random.seed(14)
    u_random14 = np.random.rand(round(Tsim/Tsc)+1, 1) - 0.5

    ## Simulation
    # Initial value
    x_Cd_vcm = np.asmatrix(np.zeros(len(A_Cd_vcm))).T
    x_Fm_vcm = np.asmatrix(np.zeros(len(A_Fm_vcm))).T
    x_Cd_pzt = np.asmatrix(np.zeros(len(A_Cd_pzt))).T
    x_Fm_pzt = np.asmatrix(np.zeros(len(A_Fm_pzt))).T
    dx_Df = np.asmatrix(np.zeros(len(A_Df))).T
    dy_Df = 0
    ddf = 0
    dx_Dp1 = np.asmatrix(np.zeros(len(A_Dp1))).T
    dy_Dp1 = 0
    dx_Dp2 = np.asmatrix(np.zeros(len(A_Dp2))).T
    dy_Dp2 = 0
    dx_Dp3 = np.asmatrix(np.zeros(len(A_Dp3))).T
    dy_Dp3 = 0
    dx_Dp4 = np.asmatrix(np.zeros(len(A_Dp4))).T
    dy_Dp4 = 0
    dx_Dp5 = np.asmatrix(np.zeros(len(A_Dp5))).T
    dy_Dp5 = 0
    dx_Dp6 = np.asmatrix(np.zeros(len(A_Dp6))).T
    dy_Dp6 = 0
    dx_Dp7 = np.asmatrix(np.zeros(len(A_Dp7))).T
    dy_Dp7 = 0
    dx_Dp8 = np.asmatrix(np.zeros(len(A_Dp8))).T
    dy_Dp8 = 0
    dx_Dp9 = np.asmatrix(np.zeros(len(A_Dp9))).T
    dy_Dp9 = 0
    dx_Dp10 = np.asmatrix(np.zeros(len(A_Dp10))).T
    dy_Dp10 = 0
    dx_Dp11 = np.asmatrix(np.zeros(len(A_Dp11))).T
    dy_Dp11 = 0
    dx_Dp12 = np.asmatrix(np.zeros(len(A_Dp12))).T
    dy_Dp12 = 0
    dx_Dp13 = np.asmatrix(np.zeros(len(A_Dp13))).T
    dy_Dp13 = 0
    ddp = 0
    dx_Pcd_vcm = np.asmatrix(np.zeros(len(A_Pcd_vcm))).T
    dy_Pcd_vcm = 0
    dx_Pcd_pzt = np.asmatrix(np.zeros(len(A_Pcd_pzt))).T
    dy_Pcd_pzt = 0
    dyc = 0
    i_RRO = 1

    # Number of inter-sampling
    nn_s = Mr_p
    nn_m = Mr_p / Mr_f

    # Simulation
    t = np.zeros(round(Tsim/Tsc)+1)
    y = np.zeros(round(Tsim/Tsc)+1)
    dRRO = np.zeros(round(Tsim/Tsc)+1)
    e = np.zeros(round(Tsim/Tsc)+1)
    ud_vcm = np.zeros(round(Tsim/Tsc)+1)
    ud_pzt = np.zeros(round(Tsim/Tsc)+1)
    uc_vcm = np.zeros(round(Tsim/Tsc)+1)
    uc_pzt = np.zeros(round(Tsim/Tsc)+1)
    df = np.zeros(round(Tsim/Tsc)+1)
    y_Pcd_vcm = np.zeros(round(Tsim/Tsc)+1)
    y_Pcd_pzt = np.zeros(round(Tsim/Tsc)+1)
    dp = np.zeros(round(Tsim/Tsc)+1)
    yc = np.zeros(round(Tsim/Tsc)+1)

    for n in range(round(Tsim/Tsc)+1):
        t[n] = Tsc*int(n-1)

        if np.mod(n,10000) == 0:
            sys.stdout.write('%s\r' % ("function_simulation: process " + str(iter) + " sim progress: " + str(100 * n/(round(Tsim/Tsc)+1))[0:4] + "%"))
            #print("\r sim progress: " + str(n) + "/" + str(round(Tsim/Tsc)+1))

        # Discrete-time system (Single-rate)
        if nn_s==Mr_p:
            nn_s=0
            y[n]=dyc

            # RRO
            dRRO[n]=RRO_data[i_RRO-1]
            if i_RRO == num_sector:
                i_RRO=1
            else:
                i_RRO=i_RRO+1

            e[n]= -y[n]+dRRO[n]
            
            # for VCM
            x_Cd_vcm = x_Cd_vcm
            y_Cd_vcm=np.dot(C_Cd_vcm,x_Cd_vcm)+D_Cd_vcm*e[n] #C_Cd_vcm*x_Cd_vcm+D_Cd_vcm*e[n]
            x_Cd_vcm=A_Cd_vcm@x_Cd_vcm+B_Cd_vcm*e[n] #A_Cd_vcm*x_Cd_vcm+B_Cd_vcm*e[n]
            ud_vcm[n]=y_Cd_vcm
            
            # for PZT
            x_Cd_pzt = x_Cd_pzt
            y_Cd_pzt=np.dot(C_Cd_pzt,x_Cd_pzt)+D_Cd_pzt*e[n] #C_Cd_pzt*x_Cd_pzt+D_Cd_pzt*e[n]
            x_Cd_pzt=A_Cd_pzt@x_Cd_pzt+B_Cd_pzt*e[n]
            ud_pzt[n]=y_Cd_pzt

        else:
            dRRO[n]=dRRO[n-1]
            e[n]=e[n-1]
            ud_vcm[n]=ud_vcm[n-1]
            ud_pzt[n]=ud_pzt[n-1]

        # Discrete-time system (Multi-rate)
        if nn_m==Mr_p/Mr_f:
            nn_m=0
        
            # for VCM
            x_Fm_vcm = x_Fm_vcm
            y_Fm_vcm=np.dot(C_Fm_vcm,x_Fm_vcm)+D_Fm_vcm*ud_vcm[n] #C_Fm_vcm*x_Fm_vcm+D_Fm_vcm*ud_vcm[n]
            x_Fm_vcm=A_Fm_vcm@x_Fm_vcm+B_Fm_vcm*ud_vcm[n]
            uc_vcm[n]=y_Fm_vcm   
        
            # for PZT
            x_Fm_pzt = x_Fm_pzt
            y_Fm_pzt= np.dot(C_Fm_pzt,x_Fm_pzt)+D_Fm_pzt*ud_pzt[n] #C_Fm_pzt*x_Fm_pzt+D_Fm_pzt*ud_pzt[n]
            x_Fm_pzt=A_Fm_pzt@x_Fm_pzt+B_Fm_pzt*ud_pzt[n]
            uc_pzt[n]=y_Fm_pzt   
        else:
            uc_vcm[n]=uc_vcm[n-1]
            uc_pzt[n]=uc_pzt[n-1]
        
        # RV disturbance
        x_Df=dx_Df
        y_Df=dy_Df
        dx_Df=A_Df@x_Df+B_Df*u_random1[n] # A_Df*x_Df+B_Df*u_random1[n]
        dy_Df= np.dot(C_Df,dx_Df) # C_Df*dx_Df
        df[n]=ddf
        ddf = dy_Df
            
        # Continuous-time system (VCM)
        x_Pcd_vcm=dx_Pcd_vcm
        y_Pcd_vcm[n]=dy_Pcd_vcm
        dx_Pcd_vcm=A_Pcd_vcm@x_Pcd_vcm+B_Pcd_vcm*(uc_vcm[n]+df[n]) # A_Pcd_vcm*x_Pcd_vcm+B_Pcd_vcm*(uc_vcm[n]+df[n])
        dy_Pcd_vcm= np.dot(C_Pcd_vcm,dx_Pcd_vcm) # C_Pcd_vcm*dx_Pcd_vcm

        # Continuous-time system (PZT)
        x_Pcd_pzt=dx_Pcd_pzt
        y_Pcd_pzt[n]=dy_Pcd_pzt
        dx_Pcd_pzt=A_Pcd_pzt@x_Pcd_pzt+B_Pcd_pzt*uc_pzt[n] # A_Pcd_pzt*x_Pcd_pzt+B_Pcd_pzt*uc_pzt[n]
        dy_Pcd_pzt= np.dot(C_Pcd_pzt,dx_Pcd_pzt) # C_Pcd_pzt*dx_Pcd_pzt

        # FAN-induced disturbance
        x_Dp1=dx_Dp1
        y_Dp1=dy_Dp1
        dx_Dp1=A_Dp1@x_Dp1+B_Dp1*u_random2[n] # A_Dp1*x_Dp1+B_Dp1*u_random2[n]
        dy_Dp1= np.dot(C_Dp1,dx_Dp1) # C_Dp1*dx_Dp1
        x_Dp2=dx_Dp2
        y_Dp2=dy_Dp2
        dx_Dp2=A_Dp2@x_Dp2+B_Dp2*u_random3[n] # A_Dp2*x_Dp2+B_Dp2*u_random3[n]
        dy_Dp2= np.dot(C_Dp2,dx_Dp2) # C_Dp2*dx_Dp2
        x_Dp3=dx_Dp3
        y_Dp3=dy_Dp3
        dx_Dp3=A_Dp3@x_Dp3+B_Dp3*u_random4[n] # A_Dp3*x_Dp3+B_Dp3*u_random4[n]
        dy_Dp3= np.dot(C_Dp3,dx_Dp3) # C_Dp3*dx_Dp3
        x_Dp4=dx_Dp4
        y_Dp4=dy_Dp4
        dx_Dp4=A_Dp4@x_Dp4+B_Dp4*u_random5[n]
        dy_Dp4= np.dot(C_Dp4,dx_Dp4)
        x_Dp5=dx_Dp5
        y_Dp5=dy_Dp5
        dx_Dp5=A_Dp5@x_Dp5+B_Dp5*u_random6[n]
        dy_Dp5=np.dot(C_Dp5,dx_Dp5)
        x_Dp6=dx_Dp6
        y_Dp6=dy_Dp6
        dx_Dp6=A_Dp6@x_Dp6+B_Dp6*u_random7[n]
        dy_Dp6=np.dot(C_Dp6,dx_Dp6)
        x_Dp7=dx_Dp7
        y_Dp7=dy_Dp7
        dx_Dp7=A_Dp7@x_Dp7+B_Dp7*u_random8[n]
        dy_Dp7=np.dot(C_Dp7,dx_Dp7)
        x_Dp8=dx_Dp8
        y_Dp8=dy_Dp8
        dx_Dp8=A_Dp8@x_Dp8+B_Dp8*u_random9[n]
        dy_Dp8=np.dot(C_Dp8,dx_Dp8)
        x_Dp9=dx_Dp9
        y_Dp9=dy_Dp9
        dx_Dp9=A_Dp9@x_Dp9+B_Dp9*u_random10[n]
        dy_Dp9=np.dot(C_Dp9,dx_Dp9)
        x_Dp10=dx_Dp10
        y_Dp10=dy_Dp10
        dx_Dp10=A_Dp10@x_Dp10+B_Dp10*u_random11[n]
        dy_Dp10=np.dot(C_Dp10,dx_Dp10)
        x_Dp11=dx_Dp11
        y_Dp11=dy_Dp11
        dx_Dp11=A_Dp11@x_Dp11+B_Dp11*u_random12[n]
        dy_Dp11=np.dot(C_Dp11,dx_Dp11)
        x_Dp12=dx_Dp12
        y_Dp12=dy_Dp12
        dx_Dp12=A_Dp12@x_Dp12+B_Dp12*u_random13[n]
        dy_Dp12=np.dot(C_Dp12,dx_Dp12)
        x_Dp13=dx_Dp13
        y_Dp13=dy_Dp13
        dx_Dp13=A_Dp13@x_Dp13+B_Dp13*u_random14[n]
        dy_Dp13=np.dot(C_Dp13,dx_Dp13)
        dp[n]=ddp
        ddp = dy_Dp1+dy_Dp2+dy_Dp3+dy_Dp4+dy_Dp5+dy_Dp6+dy_Dp7+dy_Dp8+dy_Dp9+dy_Dp10+dy_Dp11+dy_Dp12+dy_Dp13

        # Magnetic-head position
        yc[n]=dyc
        dyc=dy_Pcd_vcm+dy_Pcd_pzt+ddp
        
        nn_s=nn_s+1
        nn_m=nn_m+1

    # Remove transient response. if this gives errors, throw a '-1' at the end of each *120
    Nt=np.argmax(t >= 0.1)
    sim_result['time'] = t[Nt:Nt + Mr_p * num_sector * 120] - t[Nt]
    sim_result['uc_vcm'] = uc_vcm[Nt:Nt + Mr_p * num_sector * 120]
    sim_result['uc_pzt'] = uc_pzt[Nt:Nt + Mr_p * num_sector * 120]
    sim_result['yc'] = yc[Nt:Nt + Mr_p * num_sector * 120]
    sim_result['yc_pzt'] = y_Pcd_pzt[Nt:Nt + Mr_p * num_sector * 120]
    sim_result['dp'] = dp[Nt:Nt + Mr_p * num_sector * 120]
    sim_result['df'] = df[Nt:Nt + Mr_p * num_sector * 120]
    sim_result['dRRO'] = dRRO[Nt:Nt + Mr_p * num_sector * 120]

    # DFT
    sim_result['freq'] = np.arange(0, len(sim_result['time']))  / sim_result['time'][-1]
    sim_result['Fr_yc'] = np.fft.fft(sim_result['yc'])
    sim_result['Fr_yc_pzt'] = np.fft.fft(sim_result['yc_pzt'])
    sim_result['Fr_dp'] = np.fft.fft(sim_result['dp'])
    sim_result['Fr_df'] = np.fft.fft(sim_result['df'])

    output_path = utils.get_sim_path("res"+str(iter)+".pkl")

    with open(output_path, "wb") as outfile:
         pickle.dump(sim_result, outfile)
    
    return sim_result

if __name__ == '__main__':

    res1 = multiprocessing.Process(target=Function_simulation, name="res1", args=(plant.Sys_Pc_vcm_c1,plant.Sys_Pc_pzt_c1,utils.get_Sys_Cd_vcm(),utils.get_Sys_Fm_vcm(),utils.get_Sys_Cd_pzt(),utils.get_Sys_Fm_pzt(),plant.Ts,plant.Mr_f,1))
    res2 = multiprocessing.Process(target=Function_simulation, name="res2", args=(plant.Sys_Pc_vcm_c2,plant.Sys_Pc_pzt_c2,utils.get_Sys_Cd_vcm(),utils.get_Sys_Fm_vcm(),utils.get_Sys_Cd_pzt(),utils.get_Sys_Fm_pzt(),plant.Ts,plant.Mr_f,2))
    res3 = multiprocessing.Process(target=Function_simulation, name="res3", args=(plant.Sys_Pc_vcm_c3,plant.Sys_Pc_pzt_c3,utils.get_Sys_Cd_vcm(),utils.get_Sys_Fm_vcm(),utils.get_Sys_Cd_pzt(),utils.get_Sys_Fm_pzt(),plant.Ts,plant.Mr_f,3))
    res4 = multiprocessing.Process(target=Function_simulation, name="res4", args=(plant.Sys_Pc_vcm_c4,plant.Sys_Pc_pzt_c4,utils.get_Sys_Cd_vcm(),utils.get_Sys_Fm_vcm(),utils.get_Sys_Cd_pzt(),utils.get_Sys_Fm_pzt(),plant.Ts,plant.Mr_f,4))
    res5 = multiprocessing.Process(target=Function_simulation, name="res5", args=(plant.Sys_Pc_vcm_c5,plant.Sys_Pc_pzt_c5,utils.get_Sys_Cd_vcm(),utils.get_Sys_Fm_vcm(),utils.get_Sys_Cd_pzt(),utils.get_Sys_Fm_pzt(),plant.Ts,plant.Mr_f,5))
    res6 = multiprocessing.Process(target=Function_simulation, name="res6", args=(plant.Sys_Pc_vcm_c6,plant.Sys_Pc_pzt_c6,utils.get_Sys_Cd_vcm(),utils.get_Sys_Fm_vcm(),utils.get_Sys_Cd_pzt(),utils.get_Sys_Fm_pzt(),plant.Ts,plant.Mr_f,6))
    res7 = multiprocessing.Process(target=Function_simulation, name="res7", args=(plant.Sys_Pc_vcm_c7,plant.Sys_Pc_pzt_c7,utils.get_Sys_Cd_vcm(),utils.get_Sys_Fm_vcm(),utils.get_Sys_Cd_pzt(),utils.get_Sys_Fm_pzt(),plant.Ts,plant.Mr_f,7))
    res8 = multiprocessing.Process(target=Function_simulation, name="res8", args=(plant.Sys_Pc_vcm_c8,plant.Sys_Pc_pzt_c8,utils.get_Sys_Cd_vcm(),utils.get_Sys_Fm_vcm(),utils.get_Sys_Cd_pzt(),utils.get_Sys_Fm_pzt(),plant.Ts,plant.Mr_f,8))
    res9 = multiprocessing.Process(target=Function_simulation, name="res9", args=(plant.Sys_Pc_vcm_c9,plant.Sys_Pc_pzt_c9,utils.get_Sys_Cd_vcm(),utils.get_Sys_Fm_vcm(),utils.get_Sys_Cd_pzt(),utils.get_Sys_Fm_pzt(),plant.Ts,plant.Mr_f,9))

    res1.start()
    res2.start()
    res3.start()
    res4.start()
    res5.start()
    res6.start()
    res7.start()
    res8.start()
    res9.start()
    print("function_simulation: all tasks started")
    res1.join()
    res2.join()
    res3.join()
    res4.join()
    res5.join()
    res6.join()
    res7.join()
    res8.join()
    res9.join()
    print("function_simulation: all tasks finished")

