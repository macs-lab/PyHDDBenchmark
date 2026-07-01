import cmath
import numpy as np
import control as co
from scipy import signal, io
import hdf5storage
import json
# from pdb import set_trace as pdb
from utils import *
#from Tools import *
from control import matlab
from reduce_order_compare import *

# ------------------------------plant.m------------------------------------------
# reduce order
order = 6


#  Sampling time and Multi-rate number
num_sector = 420                   # Number of sector
num_rpm = 7200                     # Number of RPM
Ts = 1/(num_rpm/60*num_sector)   # Sampling time
Mr_f = 2                           # Multi-rate number

# Plant parameter
# VCM
Kp_vcm=3.7976e+07
omega_vcm = list(np.array([0, 5300 ,6100 ,6500 ,8050 ,9600 ,14800 ,17400 ,21000 ,26000 ,26600 ,29000 ,32200 ,38300 ,43300 ,44800])*2*np.pi)
kappa_vcm = [1, -1.0 ,+0.1 ,-0.1 ,0.04 ,-0.7 ,-0.2  ,-1.0  ,+3.0  ,-3.2  ,2.1   ,-1.5  ,+2.0  ,-0.2  ,+0.3  ,-0.5 ]
zeta_vcm = [0, 0.02 ,0.04 ,0.02 ,0.01 ,0.03 ,0.01  ,0.02  ,0.02  ,0.012 ,0.007 ,0.01  ,0.03  ,0.01  ,0.01  ,0.01 ]

# PZT
omega_pzt = list(np.array([14800 ,21500 ,28000 ,40200 ,42050,44400,46500 ,100000])*2*np.pi)
kappa_pzt = [-0.005,-0.01 ,-0.1  ,+0.8  ,0.3  ,-0.25  ,0.3  ,10.0 ]
zeta_pzt = [0.025 ,0.03  ,0.05  ,0.008  ,0.008 ,0.01 ,0.02  ,0.3 ]


# LT (case 1)
# VCM
Sys_Pc_vcm_c1 = 0
for i in range(order):
    Sys_Pc_vcm_c1_i = co.tf2ss(co.tf([0, 0, kappa_vcm[i]*Kp_vcm], [1, 2*zeta_vcm[i]*0.8*omega_vcm[i]*1.04, (omega_vcm[i]*1.04)**2]))
    Sys_Pc_vcm_c1 =  Sys_Pc_vcm_c1+Sys_Pc_vcm_c1_i

#  PZT
Sys_Pc_pzt_c1=0
for i in range(order):
    Sys_Pc_pzt_c1_i = co.tf2ss(co.tf([0, 0, kappa_pzt[i]], 
                                    [1, 2*zeta_pzt[i]*0.8*omega_pzt[i]*1.06, (omega_pzt[i]*1.06)**2]))
    Sys_Pc_pzt_c1 = Sys_Pc_pzt_c1 + Sys_Pc_pzt_c1_i
Sys_Pc_pzt_c1_ss = [Sys_Pc_pzt_c1.A, Sys_Pc_pzt_c1.B, Sys_Pc_pzt_c1.C, Sys_Pc_pzt_c1.D] 	
_, pzt_c1_freqresp = signal.freqresp(Sys_Pc_pzt_c1_ss, np.array([0.]))
Sys_Pc_pzt_c1 = Sys_Pc_pzt_c1/abs(pzt_c1_freqresp)


# RT (Case 2)
# VCM
Sys_Pc_vcm_c2 = 0
for i in range(order):
    Sys_Pc_vcm_c2_i = co.tf2ss(co.tf([0, 0, kappa_vcm[i]*Kp_vcm], 
                                    [1, 2*zeta_vcm[i]*omega_vcm[i], omega_vcm[i]**2]))
    Sys_Pc_vcm_c2 = Sys_Pc_vcm_c2 + Sys_Pc_vcm_c2_i

# PZT
Sys_Pc_pzt_c2=0
for i in range(order):
    Sys_Pc_pzt_c2_i = co.tf2ss(co.tf([0, 0, kappa_pzt[i]], 
                                [1, 2*zeta_pzt[i]*omega_pzt[i], omega_pzt[i]**2]))
    Sys_Pc_pzt_c2 = Sys_Pc_pzt_c2 + Sys_Pc_pzt_c2_i

Sys_Pc_pzt_c2_ss = [Sys_Pc_pzt_c2.A, Sys_Pc_pzt_c2.B, Sys_Pc_pzt_c2.C, Sys_Pc_pzt_c2.D] 	
_, pzt_c2_freqresp = signal.freqresp(Sys_Pc_pzt_c2_ss, np.array([0.]))
Sys_Pc_pzt_c2 = Sys_Pc_pzt_c2/abs(pzt_c2_freqresp)


# HT (Case 3)
# VCM
Sys_Pc_vcm_c3=0
for i in range(6):
    Sys_Pc_vcm_c3_i = co.tf2ss(co.tf([0, 0, kappa_vcm[i]*Kp_vcm], 
                                    [1, 2*zeta_vcm[i]*1.2*omega_vcm[i]*0.96, (omega_vcm[i]*0.96)**2]))
    Sys_Pc_vcm_c3 = Sys_Pc_vcm_c3 + Sys_Pc_vcm_c3_i

# PZT
Sys_Pc_pzt_c3=0
for i in range(6):
    Sys_Pc_pzt_c3_i = co.tf2ss(co.tf([0, 0, kappa_pzt[i]], 
                                    [1, 2*zeta_pzt[i]*1.2*omega_pzt[i]*0.94, (omega_pzt[i]*0.94)**2]))
    Sys_Pc_pzt_c3 = Sys_Pc_pzt_c3 + Sys_Pc_pzt_c3_i
Sys_Pc_pzt_c3_ss = [Sys_Pc_pzt_c3.A, Sys_Pc_pzt_c3.B, Sys_Pc_pzt_c3.C, Sys_Pc_pzt_c3.D] 	
_, pzt_c3_freqresp = signal.freqresp(Sys_Pc_pzt_c3_ss, np.array([0.]))
Sys_Pc_pzt_c3 = Sys_Pc_pzt_c3/abs(pzt_c3_freqresp)


# LT / PZT gain +5% (Case 4)
Sys_Pc_vcm_c4 = Sys_Pc_vcm_c1
Sys_Pc_pzt_c4 = Sys_Pc_pzt_c1*1.05


# RT / PZT gain +5% (Case 5)
Sys_Pc_vcm_c5 = Sys_Pc_vcm_c2
Sys_Pc_pzt_c5 = Sys_Pc_pzt_c2*1.05


# HT / PZT gain +5% (Case 6)
Sys_Pc_vcm_c6 = Sys_Pc_vcm_c3
Sys_Pc_pzt_c6 = Sys_Pc_pzt_c3*1.05


# LT / PZT gain -5% (Case 7)
Sys_Pc_vcm_c7 = Sys_Pc_vcm_c1
Sys_Pc_pzt_c7 = Sys_Pc_pzt_c1*0.95


# RT / PZT gain -5% (Case 8)
Sys_Pc_vcm_c8 = Sys_Pc_vcm_c2
Sys_Pc_pzt_c8 = Sys_Pc_pzt_c2*0.95


# HT / PZT gain -5% (Case 9)
Sys_Pc_vcm_c9 = Sys_Pc_vcm_c3
Sys_Pc_pzt_c9 = Sys_Pc_pzt_c3*0.95


Sys_Pc_vcm_all = [Sys_Pc_vcm_c1, Sys_Pc_vcm_c2, Sys_Pc_vcm_c3, 
                Sys_Pc_vcm_c4, Sys_Pc_vcm_c5, Sys_Pc_vcm_c6, 
                Sys_Pc_vcm_c7, Sys_Pc_vcm_c8, Sys_Pc_vcm_c9]

Sys_Pc_pzt_all = [Sys_Pc_pzt_c1, Sys_Pc_pzt_c2, Sys_Pc_pzt_c3, 
                Sys_Pc_pzt_c4, Sys_Pc_pzt_c5, Sys_Pc_pzt_c6, 
                Sys_Pc_pzt_c7, Sys_Pc_pzt_c8, Sys_Pc_pzt_c9]


# ------------------------------plot-control-system.m------------------------------------------

Sys_Cd_pzt = get_Sys_Cd_pzt()
Sys_Cd_vcm = get_Sys_Cd_vcm()
Sys_Fm_vcm = get_Sys_Fm_vcm()
Sys_Fm_pzt = get_Sys_Fm_pzt()

# Cotrolled object (Discrete-time system)

# Case 1
# pdb()
Sys_Pdm0_vcm_c1 = matlab.c2d(Sys_Pc_vcm_c1, Ts/Mr_f, 'zoh')
Sys_Pdm_vcm_c1 = Sys_Pdm0_vcm_c1*Sys_Fm_vcm
Sys_Pd_vcm_c1 = dts_resampling(Sys_Pdm_vcm_c1, Mr_f)

Sys_Pdm0_pzt_c1 = matlab.c2d(Sys_Pc_pzt_c1, Ts/Mr_f, 'zoh')
Sys_Pdm_pzt_c1 = Sys_Pdm0_pzt_c1*Sys_Fm_pzt
Sys_Pd_pzt_c1 = dts_resampling(Sys_Pdm_pzt_c1, Mr_f)

# Case 2
Sys_Pdm0_vcm_c2 = matlab.c2d(Sys_Pc_vcm_c2, Ts/Mr_f, 'zoh')
Sys_Pdm_vcm_c2 = Sys_Pdm0_vcm_c2*Sys_Fm_vcm
Sys_Pd_vcm_c2 = dts_resampling(Sys_Pdm_vcm_c2, Mr_f) 

Sys_Pdm0_pzt_c2 = matlab.c2d(Sys_Pc_pzt_c2, Ts/Mr_f, 'zoh')
Sys_Pdm_pzt_c2 = Sys_Pdm0_pzt_c2*Sys_Fm_pzt
Sys_Pd_pzt_c2 = dts_resampling(Sys_Pdm_pzt_c2, Mr_f)

# Case 3
Sys_Pdm0_vcm_c3 = matlab.c2d(Sys_Pc_vcm_c3, Ts/Mr_f, 'zoh')
Sys_Pdm_vcm_c3 = Sys_Pdm0_vcm_c3*Sys_Fm_vcm
Sys_Pd_vcm_c3 = dts_resampling(Sys_Pdm_vcm_c3, Mr_f)

Sys_Pdm0_pzt_c3 = matlab.c2d(Sys_Pc_pzt_c3, Ts/Mr_f, 'zoh')
Sys_Pdm_pzt_c3 = Sys_Pdm0_pzt_c3*Sys_Fm_pzt
Sys_Pd_pzt_c3 = dts_resampling(Sys_Pdm_pzt_c3, Mr_f)

# Case4
Sys_Pdm0_vcm_c4 = matlab.c2d(Sys_Pc_vcm_c4, Ts/Mr_f, 'zoh')
Sys_Pdm_vcm_c4 = Sys_Pdm0_vcm_c4*Sys_Fm_vcm
Sys_Pd_vcm_c4 = dts_resampling(Sys_Pdm_vcm_c4, Mr_f)

Sys_Pdm0_pzt_c4 = matlab.c2d(Sys_Pc_pzt_c4, Ts/Mr_f, 'zoh')
Sys_Pdm_pzt_c4 = Sys_Pdm0_pzt_c4*Sys_Fm_pzt
Sys_Pd_pzt_c4 = dts_resampling(Sys_Pdm_pzt_c4, Mr_f)

# Case 5
Sys_Pdm0_vcm_c5 = matlab.c2d(Sys_Pc_vcm_c5, Ts/Mr_f, 'zoh')
Sys_Pdm_vcm_c5 = Sys_Pdm0_vcm_c5*Sys_Fm_vcm
Sys_Pd_vcm_c5 = dts_resampling(Sys_Pdm_vcm_c5, Mr_f)

Sys_Pdm0_pzt_c5 = matlab.c2d(Sys_Pc_pzt_c5, Ts/Mr_f, 'zoh')
Sys_Pdm_pzt_c5 = Sys_Pdm0_pzt_c5*Sys_Fm_pzt
Sys_Pd_pzt_c5 = dts_resampling(Sys_Pdm_pzt_c5, Mr_f)

# Case 6
Sys_Pdm0_vcm_c6 = matlab.c2d(Sys_Pc_vcm_c6, Ts/Mr_f, 'zoh')
Sys_Pdm_vcm_c6 = Sys_Pdm0_vcm_c6*Sys_Fm_vcm
Sys_Pd_vcm_c6 = dts_resampling(Sys_Pdm_vcm_c6, Mr_f)

Sys_Pdm0_pzt_c6 = matlab.c2d(Sys_Pc_pzt_c6, Ts/Mr_f, 'zoh')
Sys_Pdm_pzt_c6 = Sys_Pdm0_pzt_c6*Sys_Fm_pzt
Sys_Pd_pzt_c6 = dts_resampling(Sys_Pdm_pzt_c6, Mr_f)

# Case 7
Sys_Pdm0_vcm_c7 = matlab.c2d(Sys_Pc_vcm_c7, Ts/Mr_f, 'zoh')
Sys_Pdm_vcm_c7 = Sys_Pdm0_vcm_c7*Sys_Fm_vcm
Sys_Pd_vcm_c7 = dts_resampling(Sys_Pdm_vcm_c7, Mr_f)

Sys_Pdm0_pzt_c7 = matlab.c2d(Sys_Pc_pzt_c7, Ts/Mr_f, 'zoh')
Sys_Pdm_pzt_c7 = Sys_Pdm0_pzt_c7*Sys_Fm_pzt
Sys_Pd_pzt_c7 = dts_resampling(Sys_Pdm_pzt_c7, Mr_f)

# Case 8
Sys_Pdm0_vcm_c8 = matlab.c2d(Sys_Pc_vcm_c8, Ts/Mr_f, 'zoh')
Sys_Pdm_vcm_c8 = Sys_Pdm0_vcm_c8*Sys_Fm_vcm
Sys_Pd_vcm_c8 = dts_resampling(Sys_Pdm_vcm_c8, Mr_f)
Sys_Pdm0_pzt_c8 = matlab.c2d(Sys_Pc_pzt_c8, Ts/Mr_f, 'zoh')
Sys_Pdm_pzt_c8 = Sys_Pdm0_pzt_c8*Sys_Fm_pzt
Sys_Pd_pzt_c8 = dts_resampling(Sys_Pdm_pzt_c8, Mr_f)

# Case 9
Sys_Pdm0_vcm_c9 = matlab.c2d(Sys_Pc_vcm_c9, Ts/Mr_f, 'zoh')
Sys_Pdm_vcm_c9 = Sys_Pdm0_vcm_c9*Sys_Fm_vcm
Sys_Pd_vcm_c9 = dts_resampling(Sys_Pdm_vcm_c9, Mr_f)

Sys_Pdm0_pzt_c9 = matlab.c2d(Sys_Pc_pzt_c9, Ts/Mr_f, 'zoh')
Sys_Pdm_pzt_c9 = Sys_Pdm0_pzt_c9*Sys_Fm_pzt
Sys_Pd_pzt_c9 = dts_resampling(Sys_Pdm_pzt_c9, Mr_f)


# All
Sys_Pd_vcm_all=[Sys_Pd_vcm_c1, Sys_Pd_vcm_c2, Sys_Pd_vcm_c3, 
                Sys_Pd_vcm_c4, Sys_Pd_vcm_c5, Sys_Pd_vcm_c6, 
                Sys_Pd_vcm_c7, Sys_Pd_vcm_c8, Sys_Pd_vcm_c9]
Sys_Pd_pzt_all=[Sys_Pd_pzt_c1, Sys_Pd_pzt_c2, Sys_Pd_pzt_c3, 
                Sys_Pd_pzt_c4, Sys_Pd_pzt_c5, Sys_Pd_pzt_c6, 
                Sys_Pd_pzt_c7, Sys_Pd_pzt_c8, Sys_Pd_pzt_c9]


# Frequency response

f = np.logspace(1, np.log10(60000), 3000)

Fr_Pc_vcm_all_reduce = freqresp(Sys_Pc_vcm_all, f*2*np.pi) 
Fr_Pc_pzt_all_reduce = freqresp(Sys_Pc_pzt_all, f*2*np.pi)
Fr_Pd_vcm_all_reduce = freqresp(Sys_Pd_vcm_all, f*2*np.pi)
Fr_Pd_pzt_all_reduce = freqresp(Sys_Pd_pzt_all, f*2*np.pi)
Fr_Cd_vcm_reduce = freqresp(Sys_Cd_vcm,f*2*np.pi)
Fr_Cd_pzt_reduce = freqresp(Sys_Cd_pzt,f*2*np.pi)
Fr_Fm_vcm_reduce = freqresp(Sys_Fm_vcm,f*2*np.pi)
Fr_Fm_pzt_reduce = freqresp(Sys_Fm_pzt,f*2*np.pi)

Fr_L_vcm_all_reduce=Fr_Pd_vcm_all_reduce*Fr_Cd_vcm_reduce
Fr_L_pzt_all_reduce=Fr_Pd_pzt_all_reduce*Fr_Cd_pzt_reduce

Fr_L_reduce=Fr_L_vcm_all_reduce+Fr_L_pzt_all_reduce
Fr_S_reduce=1./(1+Fr_L_reduce)

# Save the frequency Response
Fr_Resp_reduce = {'Fr_Pc_vcm_all_mag': abs(Fr_Pc_vcm_all_reduce).tolist(), 
           'Fr_Pc_vcm_all_phase': np.angle(Fr_Pc_vcm_all_reduce).tolist(), 
           'Fr_Pc_pzt_all_mag': abs(Fr_Pc_pzt_all_reduce).tolist(), 
           'Fr_Pc_pzt_all_phase': np.angle(Fr_Pc_pzt_all_reduce).tolist(), 
           'Fr_Pd_vcm_all_mag': abs(Fr_Pd_vcm_all_reduce).tolist(), 
           'Fr_Pd_vcm_all_phase': np.angle(Fr_Pd_vcm_all_reduce).tolist(), 
           'Fr_Pd_pzt_all_mag': abs(Fr_Pd_pzt_all_reduce).tolist(), 
           'Fr_Pd_pzt_all_phase': np.angle(Fr_Pd_pzt_all_reduce).tolist(), 
           'Fr_Cd_vcm_mag': abs(Fr_Cd_vcm_reduce).tolist(), 
           'Fr_Cd_vcm_phase': np.angle(Fr_Cd_vcm_reduce).tolist(),
           'Fr_Cd_pzt_mag': abs(Fr_Cd_pzt_reduce).tolist(), 
           'Fr_Cd_pzt_phase': np.angle(Fr_Cd_pzt_reduce).tolist(),
           'Fr_Fm_vcm_mag': abs(Fr_Fm_vcm_reduce).tolist(), 
           'Fr_Fm_vcm_phase': np.angle(Fr_Fm_vcm_reduce).tolist(),
           'Fr_Fm_pzt_mag': abs(Fr_Fm_pzt_reduce).tolist(), 
           'Fr_Fm_pzt_phase': np.angle(Fr_Fm_pzt_reduce).tolist()}

with open('Fre_Resp_reduced.json', 'w') as file:
    json.dump(Fr_Resp_reduce, file)


# Get full-order system response
file_name = './simulation_result/Fre_Resp.json'

Fr_Resp_Type = ['Fr_Pc_vcm_all', 
                'Fr_Pc_pzt_all', 
                'Fr_Pd_vcm_all', 
                'Fr_Pd_pzt_all', 
                'Fr_Cd_vcm', 
                'Fr_Cd_pzt', 
                'Fr_Fm_vcm', 
                'Fr_Fm_pzt']
Fr_Resp_all = get_Freq_Resp(file_name, Fr_Resp_Type)

Fr_Pc_vcm_all = Fr_Resp_all['Fr_Pc_vcm_all']
Fr_Pc_pzt_all = Fr_Resp_all['Fr_Pc_pzt_all']
Fr_Pd_vcm_all = Fr_Resp_all['Fr_Pd_vcm_all']
Fr_Pd_pzt_all = Fr_Resp_all['Fr_Pd_pzt_all']
Fr_Cd_vcm = Fr_Resp_all['Fr_Cd_vcm']
Fr_Cd_pzt = Fr_Resp_all['Fr_Cd_pzt']
Fr_Fm_vcm = Fr_Resp_all['Fr_Fm_vcm']
Fr_Fm_pzt = Fr_Resp_all['Fr_Fm_pzt']

Fr_L_vcm_all = Fr_Pd_vcm_all * Fr_Cd_vcm
Fr_L_pzt_all = Fr_Pd_pzt_all * Fr_Cd_pzt

Fr_L = Fr_L_vcm_all + Fr_L_pzt_all
Fr_S = 1. / (1 + Fr_L)
# Plot the Frequency Response of the system 

# Fr_Pc_vcm_all
save_path = './plot_result_ReducedOrder/figure9_The_Frequency_Response_of_Pc_vcm.png'
title = 'The Frequency Response of $Pc_{vcm}$ (Reduced Order)' 
Fr_Pc_vcm_all_mag_reduced = 20*np.log10(abs(Fr_Pc_vcm_all_reduce).T)
Fr_Pc_vcm_all_phase_reduced = 180*(np.angle(Fr_Pc_vcm_all_reduce).T)/np.pi - 180
Freq_Resp_Plot(Fr_Pc_vcm_all_mag_reduced, Fr_Pc_vcm_all_phase_reduced, f, title, (-360,0), save_path)

# Fr_Pc_pzt_all
save_path = './plot_result_ReducedOrder/figure10_The_Frequency_Response_of_Pc_pzt.png'
title = 'The Frequency Response of $Pc_{pzt}$ (Reduced Order)'
Fr_Pc_pzt_all_mag_reduce = 20*np.log10(abs(Fr_Pc_pzt_all_reduce).T)
Fr_Pc_pzt_all_phase_reduce = 180*(np.angle(Fr_Pc_pzt_all_reduce).T)/np.pi 
Freq_Resp_Plot(Fr_Pc_pzt_all_mag_reduce, Fr_Pc_pzt_all_phase_reduce, f, title, (-180,180), save_path)

# Multi_Rate_Filter
save_path = './plot_result_ReducedOrder/figure11_Multi-rate_filter.png'
Fr_Fm_vcm_mag_reduce = 20*np.log10(abs(Fr_Fm_vcm_reduce))
Fr_Fm_vcm_phase_reduce = 180*(np.angle(Fr_Fm_vcm_reduce))/np.pi
Fr_Fm_pzt_mag_reduce = 20*np.log10(abs(Fr_Fm_pzt_reduce))
Fr_Fm_pzt_phase_reduce = 180*(np.angle(Fr_Fm_pzt_reduce))/np.pi
Multi_Rate_Filter_Plot(Fr_Fm_vcm_mag_reduce, Fr_Fm_vcm_phase_reduce, 
                      Fr_Fm_pzt_mag_reduce, Fr_Fm_pzt_phase_reduce, 
                      f, 'Multi-rate Filter (Reduced Order)', save_path)

# Fr_Pd_vcm_all
save_path = './plot_result_ReducedOrder/figure12_The_Frequency_Response_of_Pd_vcm.png'
title = 'The Frequency Response of $Pd_{vcm}$ (Reduced Order)'
Fr_Pd_vcm_all_mag_reduce = 20*np.log10(abs(Fr_Pd_vcm_all_reduce).T)
Fr_Pd_vcm_all_phase_reduce = 180*(np.angle(Fr_Pd_vcm_all_reduce).T)/np.pi - 180 
Freq_Resp_Plot(Fr_Pd_vcm_all_mag_reduce, Fr_Pd_vcm_all_phase_reduce, f, title, (-360,90), save_path)

# Fr_Pd_pzt_all
save_path = './plot_result_ReducedOrder/figure13_The_Frequency_Response_of_Pd_pzt.png'
title = 'The Frequency Response of $Pd_{pzt}$ (Reduced Order)'
Fr_Pd_pzt_all_mag_reduce = 20*np.log10(abs(Fr_Pd_pzt_all_reduce).T)
Fr_Pd_pzt_all_phase_reduce = 180*(np.angle(Fr_Pd_pzt_all_reduce).T)/np.pi 
Freq_Resp_Plot(Fr_Pd_pzt_all_mag_reduce, Fr_Pd_pzt_all_phase_reduce, f, title, (-180,180), save_path)

# Fr_Cd_vcm
save_path = './plot_result_ReducedOrder/figure14_The_Frequency_Response_of_Cd_vcm.png'
title = 'The Frequency Response of $Cd_{vcm}$ (Reduced Order)'
Fr_Cd_vcm_mag_reduce = 20*np.log10(abs(Fr_Cd_vcm_reduce).T)
Fr_Cd_vcm_phase_reduce = 180*(np.angle(Fr_Cd_vcm_reduce).T)/np.pi 
Freq_Resp_Plot(Fr_Cd_vcm_mag_reduce, Fr_Cd_vcm_phase_reduce, f, title, (-180, 180), save_path)

# Fr_Cd_pzt
save_path = './plot_result_ReducedOrder/figure15_The_Frequency_Response_of_Cd_pzt.png'
title = 'The Frequency Response of $Cd_{pzt}$ (Reduced Order)'
Fr_Cd_pzt_mag_reduce = 20*np.log10(abs(Fr_Cd_pzt_reduce).T)
Fr_Cd_pzt_phase_reduce = 180*(np.angle(Fr_Cd_pzt_reduce).T)/np.pi
Freq_Resp_Plot(Fr_Cd_pzt_mag_reduce, Fr_Cd_pzt_phase_reduce, f, title, (-180, 180), save_path)

# Fr_L
save_path = './plot_result_ReducedOrder/figure16_Openloop(Bode Plot).png'
title = 'Openloop (Bode Plot, Reduced Order)' 
Fr_L_mag_reduce = 20*np.log10(abs(Fr_L_reduce).T)
Fr_L_phase_reduce = np.mod(180*(np.angle(Fr_L_reduce).T)/np.pi + 360, 360) - 360
Freq_Resp_Plot(Fr_L_mag_reduce, Fr_L_phase_reduce, f, title, (-360,0), save_path)

# Fr_L Nyquist Plot
title = 'Openloop (Nyquist Plot, Reduced Order)' 
save_path = './plot_result_ReducedOrder/figure17_Openloop(Nyquist Plot).png'
for i in range(f.shape[0]):
    if f[i] > 1/Ts/2:
        index = i
        break 
Fr_L_real_reduce = np.real(((Fr_L_reduce).T)[:, :index]).tolist()
Fr_L_imag_reduce = np.imag(((Fr_L_reduce).T)[:, :index]).tolist()
Nyquist_Plot(Fr_L_real_reduce, Fr_L_imag_reduce, title, save_path)

# Sensitive Function
save_path = './plot_result_ReducedOrder/figure18_Sensitive_Function.png'
title = 'Sensitive Function (Reduced Order)' 
Fr_S_mag_reduce = 20*np.log10(abs(Fr_S_reduce).T)
Sensitive_Function_Plot(Fr_S_mag_reduce, f, title, save_path)

plt.show()