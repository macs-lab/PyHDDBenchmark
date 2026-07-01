import control as co
from control import matlab
import numpy as np
from scipy import signal
from pdb import set_trace as pdb
import matplotlib.pyplot as plt
import json
from typing import List, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_plot_style() -> None:
    """Configure global matplotlib plotting style."""
    try:
        # Try to use seaborn if available
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        # Fallback to matplotlib's built-in style
        plt.style.use('default')
        # Configure basic style parameters
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 16,
            'font.weight': 'bold',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'axes.linewidth': 1.5,
            'axes.edgecolor': 'black',
            'xtick.major.width': 1.5,
            'ytick.major.width': 1.5,
            'xtick.minor.width': 1.5,
            'ytick.minor.width': 1.5,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'figure.titlesize': 22,
            'legend.fontsize': 14,
            'legend.frameon': True,
            'legend.edgecolor': 'black',
            'legend.framealpha': 0.8,
            'figure.figsize': [16, 12],
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
        logger.info("Using default matplotlib style with custom parameters")

def create_figure(title: str, figsize: Tuple[int, int] = (16, 12)) -> Tuple[plt.Figure, plt.Axes]:
    """Create a figure with consistent styling."""
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=22, weight='bold', pad=20)
    return fig, plt.gca()

def Freq_Resp_Plot_Compare(
    Fr_Resp_reduced_Mag: np.ndarray,
    Fr_Resp_reduced_Phase: np.ndarray,
    Fr_Resp_all_Mag: np.ndarray,
    Fr_Resp_all_Phase: np.ndarray,
    Freq: np.ndarray,
    name: str,
    phase_range: Tuple[float, float] = (-360, 90),
    save_path: Optional[str] = None
) -> None:
    """
    Compare frequency responses between reduced and full-order systems.
    
    Args:
        Fr_Resp_reduced_Mag: Magnitude of reduced-order system response
        Fr_Resp_reduced_Phase: Phase of reduced-order system response
        Fr_Resp_all_Mag: Magnitude of full-order system response
        Fr_Resp_all_Phase: Phase of full-order system response
        Freq: Frequency array
        name: Plot title
        phase_range: Phase plot range (min, max)
        save_path: Path to save the plot
    """
    setup_plot_style()
    fig, ax = plt.subplots(4, 1, sharex='col', figsize=(24, 24))
    fig.suptitle(name, fontsize=22, weight='bold')
    
    # Plot reduced-order system
    plot_frequency_response(
        ax[0], ax[1],
        Freq, Fr_Resp_reduced_Mag, Fr_Resp_reduced_Phase,
        "The result of reduced-order system",
        phase_range
    )
    
    # Plot full-order system
    plot_frequency_response(
        ax[2], ax[3],
        Freq, Fr_Resp_all_Mag, Fr_Resp_all_Phase,
        "The result of full-order system",
        phase_range
    )
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")

def plot_frequency_response(
    mag_ax: plt.Axes,
    phase_ax: plt.Axes,
    freq: np.ndarray,
    mag: np.ndarray,
    phase: np.ndarray,
    title: str,
    phase_range: Tuple[float, float]
) -> None:
    """Helper function to plot frequency response on given axes."""
    mag_ax.set_title(title, fontsize=18, weight='bold')
    mag_ax.set_ylabel("Gain [dB]")
    mag_ax.set_xscale('log')
    
    phase_ax.set_xlabel("Frequency [Hz]")
    phase_ax.set_ylabel("Phase [deg.]")
    phase_ax.set_ylim(phase_range)
    
    for i in range(mag.shape[0]):
        style = "--" if i > 5 else "-"
        mag_ax.plot(freq, mag[i], style, label=f'Case {i+1}')
        phase_ax.plot(freq, phase[i], style, label=f'Case {i+1}')
    
    if mag.shape[0] > 1:
        phase_ax.legend(
                loc="lower left", 
                prop={'family': 'Times New Roman', 'size': 16, 'weight': 'bold'}
                )

def Nyquist_Plot_Compare(
    Fr_Resp_all_real_reduce: List[List[float]],
    Fr_Resp_all_imag_reduce: List[List[float]],
    Fr_Resp_all_real: List[List[float]],
    Fr_Resp_all_imag: List[List[float]],
    title: str,
    save_path: Optional[str] = None
) -> None:
    """
    Compare Nyquist plots between reduced and full-order systems.
    
    Args:
        Fr_Resp_all_real_reduce: Real part of reduced-order system response
        Fr_Resp_all_imag_reduce: Imaginary part of reduced-order system response
        Fr_Resp_all_real: Real part of full-order system response
        Fr_Resp_all_imag: Imaginary part of full-order system response
        title: Plot title
        save_path: Path to save the plot
    """
    setup_plot_style()
    fig, ax = plt.subplots(1, 4, figsize=(48, 24))
    fig.suptitle(title, fontsize=22, weight='bold')
    
    # Plot reduced-order system
    plot_nyquist(
        ax[0], ax[1],
        Fr_Resp_all_real_reduce, Fr_Resp_all_imag_reduce,
        "Openloop (Nyquist Plot) Overall (Reduced Order)",
        "Openloop (Nyquist Plot) Detail (Reduced Order)"
    )
    
    # Plot full-order system
    plot_nyquist(
        ax[2], ax[3],
        Fr_Resp_all_real, Fr_Resp_all_imag,
        "Openloop (Nyquist Plot) Overall (Full Order)",
        "Openloop (Nyquist Plot) Detail (Full Order)"
    )
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")

def plot_nyquist(
    overall_ax: plt.Axes,
    detail_ax: plt.Axes,
    real_data: List[List[float]],
    imag_data: List[List[float]],
    overall_title: str,
    detail_title: str
) -> None:
    """Helper function to plot Nyquist plots on given axes."""
    overall_ax.set_title(overall_title, fontsize=22, weight='bold')
    overall_ax.set_xlabel("Real Axis")
    overall_ax.set_ylabel("Imaginary Axis")
    
    detail_ax.set_title(detail_title, fontsize=22, weight='bold')
    detail_ax.set_xlabel("Real Axis")
    detail_ax.set_ylabel("Imaginary Axis")
    detail_ax.set_xlim(-7, 7)
    detail_ax.set_ylim(-5, 5)
    
    for i in range(len(real_data)):
        style = "--" if i > 5 else "-"
        overall_ax.plot(real_data[i], imag_data[i], style, label=f'Case {i+1}')
        
        # Find detail view range
        d_index = next(
            (j for j in range(1, len(real_data[i]))
             if abs(real_data[i][-j]) > 7 or abs(imag_data[i][-j]) > 5),
            len(real_data[i])
        )
        
        detail_ax.plot(
            real_data[i][-d_index:],
            imag_data[i][-d_index:],
            style,
            label=f'Case {i+1}'
        )
    
    if len(real_data) > 1:
        overall_ax.legend(
               loc="lower left", 
               prop={'family': 'Times New Roman', 'size': 16, 'weight': 'bold'}
            )
        detail_ax.legend(
               loc="lower left", 
               prop={'family': 'Times New Roman', 'size': 16, 'weight': 'bold'}
            ) 
    
def Sensitive_Function_Plot_Compare(
    Fr_Resp_all_Mag_reduce: np.ndarray,
    Fr_Resp_all_Mag: np.ndarray,
    Freq: np.ndarray,
    name: str,
    save_path: Optional[str] = None
) -> None:
    """
    Compare sensitivity functions between reduced and full-order systems.
    
    Args:
        Fr_Resp_all_Mag_reduce: Magnitude of reduced-order system sensitivity
        Fr_Resp_all_Mag: Magnitude of full-order system sensitivity
        Freq: Frequency array
        name: Plot title
        save_path: Path to save the plot
    """
    setup_plot_style()
    fig, ax = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle(name, fontsize=22, weight='bold')
    
    # Plot reduced-order system
    plot_sensitivity(ax[0], Freq, Fr_Resp_all_Mag_reduce, "Reduced Order System")
    
    # Plot full-order system
    plot_sensitivity(ax[1], Freq, Fr_Resp_all_Mag, "Full Order System")
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")

def plot_sensitivity(
    ax: plt.Axes,
    freq: np.ndarray,
    mag: np.ndarray,
    title: str
) -> None:
    """Helper function to plot sensitivity function on given axes."""
    ax.set_title(title, fontsize=18, weight='bold')
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Gain [dB]")
    ax.set_xscale('log')
    
    for i in range(mag.shape[0]):
        style = "--" if i > 5 else "-"
        ax.plot(freq, mag[i], style, label=f'Case {i+1}')
    
    if mag.shape[0] > 1:
        ax.legend(
               loc="upper left", 
               prop={'family': 'Times New Roman', 'size': 16, 'weight': 'bold'}
            )
    
def Multi_Rate_Filter_Plot_Compare(
    Fr_Resp_1_Mag_reduce: np.ndarray,
    Fr_Resp_1_Phase_reduce: np.ndarray,
    Fr_Resp_2_Mag_reduce: np.ndarray,
    Fr_Resp_2_Phase_reduce: np.ndarray,
    Fr_Resp_1_Mag: np.ndarray,
    Fr_Resp_1_Phase: np.ndarray,
    Fr_Resp_2_Mag: np.ndarray,
    Fr_Resp_2_Phase: np.ndarray,
    Freq: np.ndarray,
    name: str,
    save_path: Optional[str] = None
) -> None:
    """
    Compare multi-rate filter responses between reduced and full-order systems.
    
    Args:
        Fr_Resp_1_Mag_reduce: Magnitude of first reduced-order filter response
        Fr_Resp_1_Phase_reduce: Phase of first reduced-order filter response
        Fr_Resp_2_Mag_reduce: Magnitude of second reduced-order filter response
        Fr_Resp_2_Phase_reduce: Phase of second reduced-order filter response
        Fr_Resp_1_Mag: Magnitude of first full-order filter response
        Fr_Resp_1_Phase: Phase of first full-order filter response
        Fr_Resp_2_Mag: Magnitude of second full-order filter response
        Fr_Resp_2_Phase: Phase of second full-order filter response
        Freq: Frequency array
        name: Plot title
        save_path: Path to save the plot
    """
    setup_plot_style()
    fig, ax = plt.subplots(4, 1, sharex='col', figsize=(16, 24))
    fig.suptitle(name, fontsize=22, weight='bold')
    
    # Plot reduced-order system
    plot_filter_response(
        ax[0], ax[1],
        Freq,
        Fr_Resp_1_Mag_reduce, Fr_Resp_1_Phase_reduce,
        Fr_Resp_2_Mag_reduce, Fr_Resp_2_Phase_reduce,
        "Reduced Order System"
    )
    
    # Plot full-order system
    plot_filter_response(
        ax[2], ax[3],
        Freq,
        Fr_Resp_1_Mag, Fr_Resp_1_Phase,
        Fr_Resp_2_Mag, Fr_Resp_2_Phase,
        "Full Order System"
    )
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")

def plot_filter_response(
    mag_ax: plt.Axes,
    phase_ax: plt.Axes,
    freq: np.ndarray,
    mag1: np.ndarray,
    phase1: np.ndarray,
    mag2: np.ndarray,
    phase2: np.ndarray,
    title: str
) -> None:
    """Helper function to plot filter response on given axes."""
    mag_ax.set_title(title, fontsize=18, weight='bold')
    mag_ax.set_xlabel("Frequency [Hz]")
    mag_ax.set_ylabel("Gain [dB]")
    mag_ax.set_xscale('log')
    
    phase_ax.set_xlabel("Frequency [Hz]")
    phase_ax.set_ylabel("Phase [deg.]")
    phase_ax.set_ylim(-180, 180)
    
    mag_ax.plot(freq, mag1, label='Fr_Fm_vcm')
    mag_ax.plot(freq, mag2, label='Fr_Fm_pzt')
    phase_ax.plot(freq, phase1, label='Fr_Fm_vcm')
    phase_ax.plot(freq, phase2, label='Fr_Fm_pzt')
    
    mag_ax.legend(
               loc="lower left", 
               prop={'family': 'Times New Roman', 'size': 16, 'weight': 'bold'}
               )
    phase_ax.legend(
               loc="lower left", 
               prop={'family': 'Times New Roman', 'size': 16, 'weight': 'bold'}
               )




