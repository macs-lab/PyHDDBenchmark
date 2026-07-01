import pickle
import numpy as np
import matplotlib.pyplot as plt
import utils
import plant
import sys
import print_ASCII as pa
# prints ASCII art of the system
pa.print_system()
Tp = 5.2697e-8  # 482 kTPI
Ts = plant.Ts  # get Ts from the plant parameters

PUBLICATION_FONTS = {
    'title': 18,
    'label': 16,
    'tick': 14,
    'legend': 14,
}


def apply_publication_fonts(legend=False):
    ax = plt.gca()
    ax.title.set_fontsize(PUBLICATION_FONTS['title'])
    ax.xaxis.label.set_fontsize(PUBLICATION_FONTS['label'])
    ax.yaxis.label.set_fontsize(PUBLICATION_FONTS['label'])
    ax.tick_params(axis='both', which='major',
                   labelsize=PUBLICATION_FONTS['tick'])
    if legend:
        ax.legend([f'Case {i+1}' for i in range(9)],
                  loc="lower left",
                  fontsize=PUBLICATION_FONTS['legend'])

# Simulation
# Load simulation results
try:
    sim_results = []
    for i in range(1, 10):
        sim_result = pickle.load(open(utils.get_sim_path(f"res{i}.pkl"), "rb"))
        sim_results.append(sim_result)
except Exception as e:
    print("ERROR: Simulation files not found, have you run function_simulation.py first? exiting ...")
    sys.exit()


# ============================
# Figure 1: Amplitude spectrum of df
# ============================
plt.figure(figsize=(6, 4))
plt.semilogx(sim_results[0]["freq"],
             20 * np.log10(np.abs(sim_results[0]["Fr_df"])))
plt.title('Amplitude spectrum of $d_f$')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.grid(True)
plt.xlim([10, 1 / Ts / 2])
plt.ylim(bottom=-180)
plt.tight_layout()
plt.savefig(utils.get_plot_path("figure1_Amplitude_spectrum_of_df.png"),
            dpi=600, bbox_inches='tight')
plt.savefig(utils.get_plot_path("figure1_Amplitude_spectrum_of_df.eps"),
            format='eps', bbox_inches='tight')

# ============================
# Figure 2: Amplitude spectrum of dp
# ============================
plt.figure(figsize=(6, 4))
plt.semilogx(sim_results[0]["freq"],
             20 * np.log10(np.abs(sim_results[0]["Fr_dp"])))
plt.title('Amplitude spectrum of $d_p$')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.grid(True)
plt.xlim([10, 1 / Ts / 2])
plt.ylim(bottom=-180)
plt.tight_layout()
plt.savefig(utils.get_plot_path("figure2_Amplitude_spectrum_of_dp.png"),
            dpi=600, bbox_inches='tight')
plt.savefig(utils.get_plot_path("figure2_Amplitude_spectrum_of_dp.eps"),
            format='eps', bbox_inches='tight')

# ============================
# Figure 3: dRRO (Time domain)
# ============================
plt.figure(figsize=(6, 4))
plt.plot(sim_results[0]["time"][1:420*20] * 1e3,
         sim_results[0]["dRRO"][1:420*20] * 1e9)
plt.title('$d_{RRO}$')
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude [nm]')
plt.grid(True)
plt.xlim([0, Ts * 420 * 1e3])
plt.tight_layout()
plt.savefig(utils.get_plot_path("figure3_dRRO.png"),
            dpi=600, bbox_inches='tight')
plt.savefig(utils.get_plot_path("figure3_dRRO.eps"),
            format='eps', bbox_inches='tight')

plt.show()

# ============================
# Figure 4: ypc
# ============================
plt.figure(figsize=(6, 4))
for i, sim_result in enumerate(sim_results):
    plt.plot(sim_result["time"] * 1e3,
             sim_result["yc_pzt"] * 1e9,
             '--' if i >= 7 else '-')
plt.title('$y_{cp}$')
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude [nm]')
plt.grid(True)
apply_publication_fonts(legend=True)
plt.tight_layout()
plt.savefig(utils.get_plot_path("figure4_ycp.png"),
            dpi=600, bbox_inches='tight')
plt.savefig(utils.get_plot_path("figure4_ycp.eps"),
            format='eps', bbox_inches='tight')

# ============================
# Figure 5: Max of |ycp|
# ============================
plt.figure(figsize=(6, 4))
res = np.array([sim_result["yc_pzt"] for sim_result in sim_results]).T
plt.plot(np.arange(1, 10),
         1e9 * np.max(np.abs(res), axis=0), 'o')  # ← 只保留圆点
plt.title('Max of $|y_{cp}|$')
plt.xlabel('Case number')
plt.ylabel('Value [nm]')
plt.grid(True)
apply_publication_fonts()
plt.tight_layout()
plt.savefig(utils.get_plot_path("figure5_Max_of_abs(ycp).png"),
            dpi=600, bbox_inches='tight')
plt.savefig(utils.get_plot_path("figure5_Max_of_abs(ycp).eps"),
            format='eps', bbox_inches='tight')

# ============================
# Figure 6: yc
# ============================
plt.figure(figsize=(6, 4))
for i, sim_result in enumerate(sim_results):
    plt.plot(sim_result["time"] * 1e3,
             sim_result["yc"] * 1e9,
             '--' if i >= 7 else '-')
plt.title('$y_c$')
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude [% of Track width]')
plt.grid(True)
apply_publication_fonts(legend=True)
plt.tight_layout()
plt.savefig(utils.get_plot_path("figure6_yc.png"),
            dpi=600, bbox_inches='tight')
plt.savefig(utils.get_plot_path("figure6_yc.eps"),
            format='eps', bbox_inches='tight')

# ============================
# Figure 7: Amplitude spectrum of yc
# ============================
plt.figure(figsize=(6, 4))
for i, sim_result in enumerate(sim_results):
    plt.semilogx(sim_result["freq"],
                 20 * np.log10(np.abs(sim_result["Fr_yc"])),
                 '--' if i >= 7 else '-')
plt.title('Amplitude spectrum of $y_c$')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.grid(True)
plt.xlim([10, 50e3])
apply_publication_fonts(legend=True)
plt.tight_layout()
plt.savefig(utils.get_plot_path("figure7_Amplitude_spectrum_of_yc.png"),
            dpi=600, bbox_inches='tight')
plt.savefig(utils.get_plot_path("figure7_Amplitude_spectrum_of_yc.eps"),
            format='eps', bbox_inches='tight')

# ============================
# Figure 8: 3 sigma of yc
# ============================
plt.figure(figsize=(6, 4))
res = 3 * np.std(np.array([sim_result["yc"] for sim_result in sim_results]),
                 axis=1).T
plt.plot(np.arange(1, 10), res / Tp * 100, 'o-')
plt.title('3 sigma of $y_c$')
plt.xlabel('Case number')
plt.ylabel('Value [% of Track width]')
plt.grid(True)
plt.tight_layout()
plt.savefig(utils.get_plot_path("figure8_3_sigma_of_yc.png"),
            dpi=600, bbox_inches='tight')
plt.savefig(utils.get_plot_path("figure8_3_sigma_of_yc.eps"),
            format='eps', bbox_inches='tight')

plt.show()


