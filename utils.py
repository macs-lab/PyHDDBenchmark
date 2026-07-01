import control as co
from control import matlab
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from matplotlib.ticker import MultipleLocator
import pickle




# make sure we have the output locations made
plot_output_folder_name = "plot_result"
simulation_cache_folder_name = "simulation_result"

if not os.path.exists(os.path.join(os.getcwd(), plot_output_folder_name)):
    os.makedirs(os.path.join(os.getcwd(), plot_output_folder_name))
        
if not os.path.exists(os.path.join(os.getcwd(), simulation_cache_folder_name)):
    os.makedirs(os.path.join(os.getcwd(), simulation_cache_folder_name))

def get_plot_path(fname):
    return os.path.join(os.getcwd(), os.path.join(plot_output_folder_name, fname))

def get_sim_path(fname):
    return os.path.join(os.getcwd(), os.path.join(simulation_cache_folder_name, fname))

def get_Sys_Cd_pzt():
    """
    Get the state-space representation of the PZT controller.

    Returns:
        control.StateSpace: The state-space representation of the PZT controller.
    """
    
    A = np.array([[-0.9378, 0.01655, 0.0803, 0.0885, -0.05877],
                  [0, 1.124, -0.7196, 0.3411, -0.2265],
                  [0, 0.5, 0, 0, 0],
                  [0, 0, 0, 0.5258, -0.3492],
                  [0, 0, 0, 0.5, 0]])

    B = np.array([[0.09679],
                  [0.3731],
                  [0],
                  [0.575],
                  [0]])

    C = np.array([0.4578, 0.1285, 0.6235, 0.6871, -0.4563])

    D = np.array([0.7514])

    Ts = 1.9841e-05

    return matlab.ss(A, B, C, D, Ts)


def get_Sys_Cd_vcm():
    """
    Get the state-space representation of the VCM controller.

    Returns:
        control.StateSpace: The state-space representation of the VCM controller.
    """
    
    A = np.array([[0.3698, -0.1788, 0, 0.06347, 0.07123, 0.0108, 0.1905, -0.007906, 0.3993],
                  [0.4453, 0.9419, 0, 0.02063, 0.02315, 0.003512, 0.06192, -0.00257, 0.1298],
                  [0.01809, 0.07891, 1, 0.0008384, 0.0009409, 0.0001427, 0.002516, -0.0001044, 0.005275],
                  [0, 0, 0, -0.9378, 0.06619, 0.01004, 0.177, -0.007347, 0.3711],
                  [0, 0, 0, 0, 1.124, -0.02249, 0.1706, -0.00708, 0.3576],
                  [0, 0, 0, 0, 16, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0.5258, -0.02182, 1.102],
                  [0, 0, 0, 0, 0, 0, 8, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0.2038]])

    B = np.array([[0.004101],
                  [0.001333],
                  [5.417e-05],
                  [-0.0006892],
                  [-0.0006641],
                  [0],
                  [-0.002047],
                  [0],
                  [0.008734]])

    C = np.array([-226.7, -143.1, 3.927, 62.47, 70.11, 10.63, 187.5, -7.782, 393.1])

    D = np.array([4.037])

    Ts = 1.9841e-05

    return matlab.ss(A, B, C, D, Ts)


def get_Sys_Fm_pzt():
    """
    Get the state-space representation of the PZT multi-rate filter.

    Returns:
        control.StateSpace: The state-space representation of the PZT multi-rate filter.
    """
    
    A = np.array([[1.088, -0.8294, -0.0265, 0.1441, 0.1051, 0.2489, 0.1747, 0.24, 0.06976, 0.09637],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0.4113, -0.7649, 0.115, 0.2725, 0.1913, 0.2629, 0.0764, 0.1055],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, -0.2773, -0.4853, 0.2452, 0.337, 0.09793, 0.1353],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, -0.3191, -0.1653, 0.478, 0.6604],
                  [0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, -0.7463, -0.4005],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0]])

    B = np.array([[0.08534],
                  [0],
                  [0.09346],
                  [0],
                  [0.1198],
                  [0],
                  [0.5848],
                  [0],
                  [1.],
                  [0]])

    C = np.array([-0.1256, 0.2176, -0.04971, 0.2704, 0.1971, 0.4668, 0.3277, 0.4503, 0.1309, 0.1808])

    D = np.array([0.1601])

    Ts = 9.9206e-06

    return matlab.ss(A, B, C, D, Ts)


def get_Sys_Fm_vcm():
    """
    Get the state-space representation of the VCM multi-rate filter.

    Returns:
        control.StateSpace: The state-space representation of the VCM multi-rate filter.
    """
    
    A = np.array([[1.82, -0.9386, -0.01716, 0.01955, -0.02081, 0.03475, -0.0347, 0.07612, -0.0244, 0.1173, 0.007164, 0.08666, 0.00505, 0.04144, 0.01156, 0.04172, 0.04099, 0.07872, 0.03279, 0.04083, 0.0405, 0.04084, 0.0389, 0.03774],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1.739, -0.9803, -0.01046, 0.01746, -0.01744, 0.03825, -0.01226, 0.05892, 0.0036, 0.04354, 0.002537, 0.02082, 0.005811, 0.02096, 0.0206, 0.03956, 0.01647, 0.02052, 0.02035, 0.02052, 0.01955, 0.01897],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1.175, -0.9462, -0.03552, 0.07791, -0.02498, 0.12, 0.007333, 0.0887, 0.005169, 0.04242, 0.01184, 0.04271, 0.04196, 0.08058, 0.03356, 0.04179, 0.04145, 0.04181, 0.03982, 0.03863],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0.8875, -0.8972, -0.02607, 0.1253, 0.007654, 0.09259, 0.005395, 0.04428, 0.01236, 0.04458, 0.0438, 0.08411, 0.03503, 0.04362, 0.04327, 0.04364, 0.04157, 0.04033],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0.4488, -0.6752, 0.01826, 0.2209, 0.01287, 0.1057, 0.02948, 0.1064, 0.1045, 0.2007, 0.08358, 0.1041, 0.1032, 0.1041, 0.09918, 0.09622],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0882, -0.8504, 0.00686, 0.0563, 0.01571, 0.05668, 0.05569, 0.1069, 0.04454, 0.05547, 0.05502, 0.05548, 0.05285, 0.05127],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.1566, -0.8472, 0.03353, 0.121, 0.1189, 0.2283, 0.09507, 0.1184, 0.1174, 0.1184, 0.1128, 0.1094],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.4247, -0.8346, 0.1276, 0.245, 0.102, 0.1271, 0.1261, 0.1271, 0.1211, 0.1175],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.6766, -0.6694, 0.1213, 0.1511, 0.1499, 0.1511, 0.144, 0.1397],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.29, -0.7876, 0.1644, 0.1658, 0.1579, 0.1532],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.577, -0.7635, 0.1753, 0.17],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.632, -0.7563],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  ])

    B = np.array([[0.09968],
                  [0],
                  [0.05009],
                  [0],
                  [0.102],
                  [0],
                  [0.1065],
                  [0],
                  [0.2541],
                  [0],
                  [0.1354],
                  [0],
                  [0.289],
                  [0],
                  [0.3103],
                  [0],
                  [0.3688],
                  [0],
                  [0.4047],
                  [0],
                  [0.4491],
                  [0],
                  [0.5],
                  [0]])

    C = np.array([-0.1736, 0.1852, -0.06698, 0.07632, -0.08124, 0.1356, -0.1354, 0.2971, -0.09525, 0.4577, 0.02796, 0.3382, 0.01971, 0.1618, 0.04514, 0.1628, 0.16, 0.3073, 0.128, 0.1594, 0.1581, 0.1594, 0.1519, 0.1473])

    D = np.array([0.3891])

    Ts = 9.9206e-06

    return matlab.ss(A, B, C, D, Ts)

def dts_resampling(sys, n):
    """
    Resample a discrete-time system by a factor of n.
    Args:
        sys (control.StateSpace): The discrete-time system to be resampled.
        n (int): The resampling factor.

    Returns:
        control.StateSpace: The resampled discrete-time system.
    """
    A = np.array(sys.A)
    B = np.array(sys.B)
    C = np.array(sys.C)
    D = np.array(sys.D)
    Ts = sys.dt

    Az = np.array([[i for i in j] for j in A])
    Bz = np.array([[i for i in j] for j in B])
    for i in range(1, n):
        Bz = Bz + Az@B
        Az = Az@A
    return matlab.ss(Az, Bz, C, D, Ts*n)


def freqresp(sys_list, freq):
    """
    Calculate the frequency response of a list of systems.
    Args:
        sys_list (list or control.StateSpace): A list of systems or a single system.
        freq (numpy.ndarray): The frequency points at which to evaluate the frequency response.

    Returns:
        numpy.ndarray: The frequency response of the systems.
    """

    if isinstance(sys_list, list):
        pass
    else:
        sys_list = [sys_list]
    reps_list = []

    for sys in sys_list:
        mag, phase, w = co.freqresp(sys, freq)
        reps = np.array([mag[i]*complex(np.cos(phase[i]), np.sin(phase[i])) for i in range(mag.shape[0])])
        reps_list.append(reps)

    return np.stack(reps_list, axis=1)




def get_Freq_Resp(file_name, Fr_Resp_Type):
    """
    Get the frequency response data from a JSON file.
    Args:
        file_name (str): The name of the JSON file containing the frequency response data.
        Fr_Resp_Type (list): The types of frequency responses to retrieve.

    Returns:
        dict: A dictionary containing the frequency response data.
    """
    Fr_Resp_all = {}

    with open(file_name, 'r') as f:
        Fr_Resp_Json = json.load(f)

    for Fr_Resp_Item in Fr_Resp_Type:
        Fr_Resp = []
        Fr_Resp_Mag = Fr_Resp_Json[Fr_Resp_Item + '_mag']
        Fr_Resp_Phase = Fr_Resp_Json[Fr_Resp_Item + '_phase']

        for i in range(len(Fr_Resp_Mag)):
            Fr_Resp.append([])
            for j in range(len(Fr_Resp_Mag[i])):
                mag = Fr_Resp_Mag[i][j]
                phase = Fr_Resp_Phase[i][j]
                Fr_Resp[i].append(complex(mag*np.cos(phase), mag*np.sin(phase)))
        
        Fr_Resp_all[Fr_Resp_Item] = np.array(Fr_Resp)

    return Fr_Resp_all


def Freq_Resp_Plot(mag, phase, f, title, phase_lim, save_path):
    plt.rcParams.update({'font.size': 16})  # Increase base font size
    fig = plt.figure(figsize=(12, 8))

    # Magnitude plot
    ax1 = fig.add_subplot(211)
    lines = []
    for i in range(len(mag)):
        line, = ax1.semilogx(f, mag[i])
        lines.append(line)
    ax1.grid(True)
    ax1.set_ylabel('Magnitude [dB]', fontsize=18)
    ax1.set_title(title, pad=15, fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Phase plot
    ax2 = fig.add_subplot(212)
    for i in range(len(phase)):
        ax2.semilogx(f, phase[i], color=lines[i].get_color())
    ax2.grid(True)
    ax2.set_ylabel('Phase [deg]', fontsize=18)
    ax2.set_xlabel('Frequency [Hz]', fontsize=18)
    ax2.set_ylim(phase_lim)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    # Add legend
    if len(mag) > 1:  # Only add legend if there are multiple cases
        labels = [f'Case {i+1}' for i in range(len(mag))]
        ax2.legend(lines, labels, loc='lower left', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def Nyquist_Plot(real, imag, title, save_path):
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    lines = []
    for i in range(len(real)):
        line, = ax.plot(real[i], imag[i])
        lines.append(line)
    
    ax.grid(True)
    ax.set_xlabel('Real', fontsize=18)
    ax.set_ylabel('Imaginary', fontsize=18)
    ax.set_title(title, pad=15, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Set axis limits for detail view
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 2)
    ax.set_aspect('equal')
    
    # Add legend
    if len(real) > 1:  # Only add legend if there are multiple cases
        labels = [f'Case {i+1}' for i in range(len(real))]
        ax.legend(lines, labels, loc='lower left', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def Sensitive_Function_Plot(mag, f, title, save_path):
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    
    lines = []
    for i in range(len(mag)):
        line, = ax.semilogx(f, mag[i])
        lines.append(line)
    
    ax.grid(True)
    ax.set_xlabel('Frequency [Hz]', fontsize=18)
    ax.set_ylabel('Magnitude [dB]', fontsize=18)
    ax.set_title(title, pad=15, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Add legend
    if len(mag) > 1:  # Only add legend if there are multiple cases
        labels = [f'Case {i+1}' for i in range(len(mag))]
        ax.legend(lines, labels, loc='upper left', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def Multi_Rate_Filter_Plot(mag_vcm, phase_vcm, mag_pzt, phase_pzt, f, title, save_path):
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(12, 8))
    
    # Magnitude plot
    ax1 = fig.add_subplot(211)
    ax1.semilogx(f, mag_vcm, label='VCM')
    ax1.semilogx(f, mag_pzt, label='PZT')
    ax1.grid(True)
    ax1.set_ylabel('Magnitude [dB]', fontsize=18)
    ax1.set_title(title, pad=15, fontsize=20)
    ax1.legend(fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Phase plot
    ax2 = fig.add_subplot(212)
    ax2.semilogx(f, phase_vcm, label='VCM')
    ax2.semilogx(f, phase_pzt, label='PZT')
    ax2.grid(True)
    ax2.set_ylabel('Phase [deg]', fontsize=18)
    ax2.set_xlabel('Frequency [Hz]', fontsize=18)
    ax2.legend(fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()




