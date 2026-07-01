"""
HDD Notch Filter Auto Design Tool - V2
Automated notch filter design for HDD control systems based on reinforcement learning.

V2 improvements over V1:
  1. Gap-driven notch placement:
       Instead of anchoring new notches at raw P*Fm peaks (which the existing
       multirate filter may already suppress), compute the *baseline* sensitivity
       S_baseline = 1/(1 + L_vcm + L_pzt) without any new notch.  Peaks that
       remain elevated in S_baseline are the genuine "gaps" still needing
       suppression.

  2. Channel-specific gap attribution:
       Each S_baseline peak is assigned to the VCM or PZT notch by comparing
       the partial loop gains |L_vcm(f)| vs |L_pzt(f)| at that frequency.
       This respects the natural frequency division (VCM: low, PZT: high)
       without a hard frequency cutoff.

  3. Relative objective (waterbed-aware):
       Primary goal : reduce S at each gap frequency by S_gap_improvement_target
                      (default 1.0 dB) relative to S_baseline.
       Global guard : global S_peak must not exceed S_baseline_peak +
                      S_waterbed_margin (default 0.5 dB) so the waterbed
                      effect does not silently worsen other frequencies.

  4. Smaller state dimension:
       Static plant features (30-dim P*Fm peaks) are replaced by 15-dim
       S_baseline peaks, which are more directly informative about what still
       needs fixing.

Usage:  identical CLI to V1 (see NOTCH_DESIGNER_V2.md for details).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import control.matlab as matlab
from control import freqresp
import scipy.signal as signal
from scipy.optimize import differential_evolution
import sys
import os
import copy
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import plant
import utils
import RL_config as config


# ---------------------------------------------------------------------------
# V2 helper: build the extended config dict (called instead of
# config.get_simple_config() throughout this file).
# ---------------------------------------------------------------------------
def get_simple_config_v2():
    """Extend the base simple config with V2 gap-analysis parameters."""
    env_config = config.get_simple_config()
    # How many dB each gap-frequency peak must be reduced vs S_baseline.
    env_config['S_gap_improvement_target'] = 1.0
    # How many dB above S_baseline_peak the global S_peak is still acceptable
    # (waterbed tolerance).  Loosened to 1.0 dB: 0.5 dB was too tight and
    # caused the agent to default to trivial (depth~-1 dB) notches to avoid
    # any risk of waterbed penalty.
    env_config['S_waterbed_margin'] = 1.0
    # Only S_baseline peaks above this threshold (dB) are treated as gaps worth
    # targeting with a new notch.  2 dB is a conservative choice that avoids
    # chasing trivial humps.
    env_config['S_gap_threshold'] = 2.0
    return env_config


# ---------------------------------------------------------------------------
# Unchanged helper dataclass
# ---------------------------------------------------------------------------
class NotchFilterParams:
    """Notch filter parameters"""
    def __init__(self, center_freq, bandwidth, depth):
        self.center_freq = center_freq
        self.bandwidth = bandwidth
        self.depth = depth
        self.q_factor = center_freq / bandwidth


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------
class SimpleHDDNotchDesignEnv:
    """
    HDD Notch Filter Design Environment - V2.

    Key differences from V1:
      - plant_features (15-dim) now holds S_baseline peaks instead of
        raw P*Fm peaks (was 30-dim).
      - current_vcm_peak_refs / current_pzt_peak_refs are taken from the
        gap analysis of S_baseline, not from raw plant peaks.
      - _objective_from_performance uses relative improvement vs S_baseline
        plus an explicit waterbed guard.
    """

    def __init__(self, env_config):
        self.config = env_config

        # System parameters
        try:
            self.Ts = float(plant.Ts)
        except Exception:
            self.Ts = 1.9841e-05
        try:
            self.Mr_f = int(plant.Mr_f)
        except Exception:
            self.Mr_f = 2

        # Plant cases
        self.plant_cases = [
            ('c1', plant.Sys_Pc_vcm_c1, plant.Sys_Pc_pzt_c1),
            ('c2', plant.Sys_Pc_vcm_c2, plant.Sys_Pc_pzt_c2),
            ('c3', plant.Sys_Pc_vcm_c3, plant.Sys_Pc_pzt_c3),
            ('c4', plant.Sys_Pc_vcm_c4, plant.Sys_Pc_pzt_c4),
            ('c5', plant.Sys_Pc_vcm_c5, plant.Sys_Pc_pzt_c5),
            ('c6', plant.Sys_Pc_vcm_c6, plant.Sys_Pc_pzt_c6),
            ('c7', plant.Sys_Pc_vcm_c7, plant.Sys_Pc_pzt_c7),
            ('c8', plant.Sys_Pc_vcm_c8, plant.Sys_Pc_pzt_c8),
            ('c9', plant.Sys_Pc_vcm_c9, plant.Sys_Pc_pzt_c9),
        ]

        self.current_case = None
        self.current_vcm_plant = None
        self.current_pzt_plant = None

        # Controllers
        if hasattr(utils, 'Sys_Cd_vcm'):
            self.Sys_Cd_vcm = utils.Sys_Cd_vcm
        elif hasattr(utils, 'get_Sys_Cd_vcm'):
            self.Sys_Cd_vcm = utils.get_Sys_Cd_vcm()
        else:
            raise AttributeError('No VCM controller found in utils')

        if hasattr(utils, 'Sys_Cd_pzt'):
            self.Sys_Cd_pzt = utils.Sys_Cd_pzt
        elif hasattr(utils, 'get_Sys_Cd_pzt'):
            self.Sys_Cd_pzt = utils.get_Sys_Cd_pzt()
        else:
            raise AttributeError('No PZT controller found in utils')

        # Multi-rate filters
        if hasattr(utils, 'Sys_Fm_vcm'):
            self.Sys_Fm_vcm = utils.Sys_Fm_vcm
        elif hasattr(utils, 'get_Sys_Fm_vcm'):
            self.Sys_Fm_vcm = utils.get_Sys_Fm_vcm()
        else:
            raise AttributeError('No VCM multirate filter found in utils')

        if hasattr(utils, 'Sys_Fm_pzt'):
            self.Sys_Fm_pzt = utils.Sys_Fm_pzt
        elif hasattr(utils, 'get_Sys_Fm_pzt'):
            self.Sys_Fm_pzt = utils.get_Sys_Fm_pzt()
        else:
            raise AttributeError('No PZT multirate filter found in utils')

        # Reward / target config (unchanged from V1)
        self.weights = env_config['weights']
        self.targets = env_config['targets']
        # 60: losses are now bounded (waterbed max=500, gap max=180), so a larger
        # scale keeps reward in [-10, 1] range for stable PPO learning.
        self.reward_scale = float(env_config.get('reward_scale', 60.0))
        self.frequency_delta_decades = float(env_config.get('frequency_delta_decades', 0.12))
        self.delta_max = float(env_config.get('delta_max', 0.12))
        self.notches_per_channel = int(env_config.get('notches_per_channel', 2))
        self.params_per_notch = 3
        self.param_dim = 2 * self.notches_per_channel * self.params_per_notch

        # V2: gap-analysis parameters (fall back to sensible defaults if absent)
        self.S_gap_improvement_target = float(env_config.get('S_gap_improvement_target', 1.0))
        self.S_waterbed_margin = float(env_config.get('S_waterbed_margin', 0.5))
        self.S_gap_threshold = float(env_config.get('S_gap_threshold', 2.0))

        # Frequency grid
        slow_nyquist_freq = 1.0 / self.Ts / 2.0
        max_freq = min(float(env_config.get('max_closed_loop_freq', 25000.0)), slow_nyquist_freq * 0.98)
        num_freq_points = int(env_config.get('num_freq_points', 800))
        self.freq_range = np.logspace(1, np.log10(max_freq), num_freq_points)
        self.omega = 2 * np.pi * self.freq_range

        # Precompute fixed controller / filter frequency responses
        self.Cd_vcm_fr = self._ensure_1d_fr(utils.freqresp(self.Sys_Cd_vcm, self.omega))
        self.Cd_pzt_fr = self._ensure_1d_fr(utils.freqresp(self.Sys_Cd_pzt, self.omega))
        self.Fm_vcm_fr = self._ensure_1d_fr(utils.freqresp(self.Sys_Fm_vcm, self.omega))
        self.Fm_pzt_fr = self._ensure_1d_fr(utils.freqresp(self.Sys_Fm_pzt, self.omega))

        self.param_low, self.param_high = self._load_action_bounds(env_config.get('action_bounds'))
        self.param_range = np.clip(self.param_high - self.param_low, 1e-6, None)

        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()

        # Plant response cache (computed once at startup)
        self.plant_fr_cache = {}
        self.plant_chain_cache = {}
        self._precompute_plant_responses()

        # Optional DE warm-start
        self.initial_params_by_case = None
        self.init_noise = float(env_config.get('init_noise', 0.15))

        # State variables
        self.current_notch_params = self._midpoint_params()

        # V2: plant_features is 15-dim (S_baseline peaks) instead of 30-dim.
        self.plant_features = np.zeros(15)
        self.current_system_features = np.zeros(15)  # current S peaks (dynamic)
        self.current_performance = np.array([0.0])

        self.current_vcm_peak_refs = []
        self.current_pzt_peak_refs = []

        # V2: baseline sensitivity storage (set per episode in _select_plant_case)
        self.S_baseline_fr = None          # S without any new notch
        self.S_baseline_peak = 0.0         # max dB of S_baseline
        self.current_gap_freqs = []        # gap frequencies for this episode

        # Placeholders
        self.current_vcm_fr_with_notch = None
        self.current_pzt_fr_with_notch = None
        self.current_sensitivity_fr = None

    # -----------------------------------------------------------------------
    # Observation / action spaces
    # -----------------------------------------------------------------------

    def _create_observation_space(self):
        """
        State layout (V2):
          [S_baseline_peaks (15)]        <- static reference: what still needs fixing
          [current_S_peaks   (15)]       <- dynamic: how well current notch is working
          [notch_params_norm (param_dim)]
          [sensitivity_peak  (1)]
        Total: 31 + param_dim  (was 46 + param_dim in V1)
        """
        return {
            'shape': (15 + 15 + self.param_dim + 1,),
            'low': -np.inf,
            'high': np.inf
        }

    def _create_action_space(self):
        return {
            'shape': (self.param_dim,),
            'low': np.array([-1.0] * self.param_dim),
            'high': np.array([1.0] * self.param_dim)
        }

    # -----------------------------------------------------------------------
    # Frequency-response helpers (unchanged from V1)
    # -----------------------------------------------------------------------

    def _ensure_1d_fr(self, fr):
        fr = np.asarray(fr)
        fr = np.squeeze(fr)
        if fr.ndim == 0:
            return np.array([fr], dtype=complex)
        if fr.ndim > 1:
            fr = fr.reshape(fr.shape[0], -1)[0]
        return fr

    def _freqresp_1d(self, system):
        response = utils.freqresp(system, self.omega)
        return self._ensure_1d_fr(response)

    def _ss_tuple(self, system):
        return (
            np.asarray(system.A, dtype=np.float64),
            np.asarray(system.B, dtype=np.float64),
            np.asarray(system.C, dtype=np.float64),
            np.asarray(system.D, dtype=np.float64),
        )

    def _series_ss(self, sys1, sys2):
        if sys2 is None:
            return sys1
        if sys1 is None:
            return sys2
        A1, B1, C1, D1 = sys1
        A2, B2, C2, D2 = sys2
        n1 = A1.shape[0]
        n2 = A2.shape[0]
        A = np.block([[A1, B1 @ C2], [np.zeros((n2, n1)), A2]])
        B = np.vstack([B1 @ D2, B2])
        C = np.hstack([C1, D1 @ C2])
        D = D1 @ D2
        return A, B, C, D

    def _resample_ss(self, sys_ss):
        A, B, C, D = sys_ss
        Az = A.copy()
        Bz = B.copy()
        for _ in range(1, self.Mr_f):
            Bz = Bz + Az @ B
            Az = Az @ A
        return Az, Bz, C, D

    def _freqresp_ss(self, sys_ss):
        A, B, C, D = sys_ss
        sys = signal.dlti(A, B, C, D, dt=self.Ts)
        _, response = signal.dfreqresp(sys, w=self.omega * self.Ts)
        return np.asarray(response, dtype=np.complex128)

    def _notch_ss(self, f0, bw, depth_db):
        if depth_db >= 0:
            return None
        ts = self.Ts / self.Mr_f
        w0 = 2 * np.pi * f0
        zeta = bw / (2 * f0)
        depth_lin = max(10 ** (depth_db / 20.0), 1e-4)
        num = [1.0, 2.0 * zeta * w0 * depth_lin, w0**2]
        den = [1.0, 2.0 * zeta * w0, w0**2]
        b, a, _ = signal.cont2discrete((num, den), ts, method='zoh')
        A, B, C, D = signal.tf2ss(b.ravel(), a)
        return (
            np.asarray(A, dtype=np.float64),
            np.asarray(B, dtype=np.float64),
            np.asarray(C, dtype=np.float64),
            np.asarray(D, dtype=np.float64),
        )

    def _build_notch_filter_tf(self, f0, bw, depth_db):
        sample_time = self.Ts / self.Mr_f
        if depth_db >= 0:
            return matlab.tf([1.0], [1.0], sample_time)
        w0 = 2 * np.pi * f0
        zeta = bw / (2 * f0)
        depth_lin = max(10 ** (depth_db / 20.0), 1e-4)
        num = [1, 2 * zeta * w0 * depth_lin, w0**2]
        den = [1, 2 * zeta * w0, w0**2]
        notch_ct = matlab.tf(num, den)
        return matlab.c2d(notch_ct, sample_time, 'zoh')

    def _create_digital_path(self, sys_pc, sys_fm, notch_tf=None):
        sys_pdm0 = matlab.c2d(sys_pc, self.Ts / self.Mr_f, 'zoh')
        sys_chain = sys_pdm0 * sys_fm
        if notch_tf is not None:
            sys_chain = sys_chain * notch_tf
        return utils.dts_resampling(sys_chain, self.Mr_f)

    def _create_digital_path_from_chain(self, sys_chain, notch_tf=None):
        if notch_tf is not None:
            sys_chain = sys_chain * notch_tf
        return utils.dts_resampling(sys_chain, self.Mr_f)

    # -----------------------------------------------------------------------
    # Action / parameter bounds (unchanged from V1)
    # -----------------------------------------------------------------------

    def _load_action_bounds(self, action_bounds):
        if action_bounds is None:
            low_6 = np.array([5000.0, 100.0, -60.0, 10000.0, 100.0, -60.0], dtype=np.float32)
            high_6 = np.array([45000.0, 5000.0, 0.0, 47000.0, 5000.0, 0.0], dtype=np.float32)
            low, high = self._expand_base_bounds(low_6, high_6)
            return low, high
        low = np.array(action_bounds['low'], dtype=np.float32)
        high = np.array(action_bounds['high'], dtype=np.float32)
        if low.shape == (6,) and high.shape == (6,):
            low, high = self._expand_base_bounds(low, high)
        if low.shape != (self.param_dim,) or high.shape != (self.param_dim,):
            raise ValueError(f"Action bounds must provide 6 or {self.param_dim} values.")
        return low, high

    def _expand_base_bounds(self, low_6, high_6):
        vcm_low, pzt_low = low_6[:3], low_6[3:]
        vcm_high, pzt_high = high_6[:3], high_6[3:]
        low = np.concatenate(
            [np.tile(vcm_low, self.notches_per_channel), np.tile(pzt_low, self.notches_per_channel)]
        ).astype(np.float32)
        high = np.concatenate(
            [np.tile(vcm_high, self.notches_per_channel), np.tile(pzt_high, self.notches_per_channel)]
        ).astype(np.float32)
        return low, high

    def _midpoint_params(self):
        params = np.zeros(self.param_dim)
        for i in range(self.param_dim):
            low = self.param_low[i]
            high = self.param_high[i]
            if i % 3 in (0, 1):
                params[i] = np.sqrt(low * high)
            else:
                params[i] = (low + high) / 2.0
        return params

    # -----------------------------------------------------------------------
    # Action mapping (unchanged from V1)
    # -----------------------------------------------------------------------

    def _action_to_params(self, action):
        """Map normalized action [-1,1] to physical notch parameters."""
        action = np.clip(action, self.action_space['low'], self.action_space['high'])
        normalized = (action + 1.0) / 2.0
        params = np.zeros_like(normalized, dtype=np.float64)
        for i in range(self.param_dim):
            low = self.param_low[i]
            high = self.param_high[i]
            field = i % 3
            channel_offset = 0 if i < self.notches_per_channel * 3 else self.notches_per_channel * 3
            notch_idx = (i - channel_offset) // 3
            if field == 0:
                refs = self.current_vcm_peak_refs if channel_offset == 0 else self.current_pzt_peak_refs
                fallback = np.sqrt(low * high)
                ref = refs[notch_idx] if notch_idx < len(refs) and refs[notch_idx] > 0 else fallback
                params[i] = np.clip(
                    ref * (10.0 ** (action[i] * self.frequency_delta_decades)),
                    low, high,
                )
            elif field == 1:
                log_low = np.log10(low)
                log_high = np.log10(high)
                params[i] = 10 ** (log_low + normalized[i] * (log_high - log_low))
            else:
                params[i] = low + normalized[i] * (high - low)
        return params

    def _params_to_action(self, params):
        params = np.asarray(params, dtype=np.float64)
        action = np.zeros(self.param_dim, dtype=np.float32)
        for i in range(self.param_dim):
            low = self.param_low[i]
            high = self.param_high[i]
            field = i % 3
            channel_offset = 0 if i < self.notches_per_channel * 3 else self.notches_per_channel * 3
            notch_idx = (i - channel_offset) // 3
            if field == 0:
                refs = self.current_vcm_peak_refs if channel_offset == 0 else self.current_pzt_peak_refs
                fallback = np.sqrt(low * high)
                ref = refs[notch_idx] if notch_idx < len(refs) and refs[notch_idx] > 0 else fallback
                action[i] = np.log10(np.clip(params[i], low, high) / ref) / self.frequency_delta_decades
            elif field == 1:
                log_low = np.log10(low)
                log_high = np.log10(high)
                n = (np.log10(np.clip(params[i], low, high)) - log_low) / (log_high - log_low)
                action[i] = 2.0 * n - 1.0
            else:
                n = (np.clip(params[i], low, high) - low) / (high - low)
                action[i] = 2.0 * n - 1.0
        return np.clip(action, -1.0, 1.0)

    # -----------------------------------------------------------------------
    # Plant response pre-computation (unchanged from V1)
    # -----------------------------------------------------------------------

    def _precompute_plant_responses(self):
        """Precompute full-pipeline P*Fm responses for all plant cases (no notch)."""
        print("Precomputing full-pipeline plant responses for optimization...")
        fast_ts = self.Ts / self.Mr_f
        for case_name, base_vcm, base_pzt in self.plant_cases:
            sys_pdm0_vcm = matlab.c2d(base_vcm, fast_ts, 'zoh')
            sys_chain_vcm = sys_pdm0_vcm * self.Sys_Fm_vcm
            chain_vcm_ss = self._ss_tuple(sys_chain_vcm)
            fr_vcm = self._freqresp_ss(self._resample_ss(chain_vcm_ss))

            sys_pdm0_pzt = matlab.c2d(base_pzt, fast_ts, 'zoh')
            sys_chain_pzt = sys_pdm0_pzt * self.Sys_Fm_pzt
            chain_pzt_ss = self._ss_tuple(sys_chain_pzt)
            fr_pzt = self._freqresp_ss(self._resample_ss(chain_pzt_ss))

            self.plant_fr_cache[case_name] = (fr_vcm, fr_pzt)
            self.plant_chain_cache[case_name] = (chain_vcm_ss, chain_pzt_ss)

    # -----------------------------------------------------------------------
    # DE warm-start loader (unchanged from V1)
    # -----------------------------------------------------------------------

    def load_initial_params(self, npz_path):
        """Warm-start each episode from DE-optimized notch params."""
        data = np.load(npz_path)
        params = np.asarray(data['notch_params'], dtype=np.float64)
        plant_case = str(data.get('plant_case', 'all'))
        self.initial_params_by_case = {}
        if plant_case == 'all':
            for case_name, _, _ in self.plant_cases:
                self.initial_params_by_case[case_name] = params.copy()
        else:
            self.initial_params_by_case[plant_case] = params.copy()
        print(f"Loaded DE warm-start from: {os.path.abspath(npz_path)}")
        print(f"  plant_case={plant_case}, "
              f"params={format_notch_params(params, self.notches_per_channel)}")

    # -----------------------------------------------------------------------
    # Notch filter frequency response (unchanged from V1)
    # -----------------------------------------------------------------------

    def _compute_notch_fr(self, f0, bw, depth_db):
        ts = self.Ts / self.Mr_f
        w0 = 2 * np.pi * f0
        zeta = bw / (2 * f0)
        depth_lin = max(10 ** (depth_db / 20.0), 1e-6)
        num = [1.0, 2.0 * zeta * w0 * depth_lin, w0**2]
        den = [1.0, 2.0 * zeta * w0, w0**2]
        res = signal.cont2discrete((num, den), ts, method='zoh')
        b = res[0].ravel()
        a = res[1]
        w_digital = self.omega * ts
        _, h = signal.freqz(b, a, worN=w_digital)
        return h

    # -----------------------------------------------------------------------
    # V2 - Plant case selection with baseline sensitivity and gap analysis
    # -----------------------------------------------------------------------

    def _select_plant_case(self, case_name):
        """
        Select a plant case and compute all per-episode reference data.

        V2 changes vs V1:
          - Computes S_baseline (no new notch) and stores it.
          - Identifies gap frequencies via channel attribution
            (|L_vcm| vs |L_pzt|) and uses them as notch anchors,
            replacing the raw P*Fm-peak anchors used in V1.
          - plant_features now holds 15-dim S_baseline peak features
            instead of 30-dim raw plant peaks.
        """
        self.current_case = case_name
        self.current_gain_scale = 1.0
        self.current_base_fr = self.plant_fr_cache[case_name]
        self.current_fast_chain = self.plant_chain_cache[case_name]

        self.base_vcm_fr = self.current_base_fr[0]  # P_vcm * Fm_vcm, slow rate
        self.base_pzt_fr = self.current_base_fr[1]  # P_pzt * Fm_pzt, slow rate

        # --- V2: compute baseline sensitivity without any new notch ---
        # L_vcm = P_vcm * Fm_vcm * Cd_vcm  (all at slow rate)
        # L_pzt = P_pzt * Fm_pzt * Cd_pzt
        L_vcm_base = self.base_vcm_fr * self.Cd_vcm_fr
        L_pzt_base = self.base_pzt_fr * self.Cd_pzt_fr
        L_baseline = L_vcm_base + L_pzt_base
        self.S_baseline_fr = 1.0 / (1.0 + L_baseline)
        self.S_baseline_peak = float(
            np.max(20 * np.log10(np.abs(self.S_baseline_fr) + 1e-12))
        )

        # --- V2: gap analysis - find S_baseline peaks and attribute to channels ---
        vcm_gap_freqs, pzt_gap_freqs = self._find_channel_gaps()
        self.current_gap_freqs = vcm_gap_freqs + pzt_gap_freqs

        # Use gap frequencies as the notch frequency anchors (replaces V1 plant peaks)
        self.current_vcm_peak_refs = vcm_gap_freqs
        self.current_pzt_peak_refs = pzt_gap_freqs

        # --- V2: plant_features = S_baseline peaks (15 dim, replaces 30-dim) ---
        self.plant_features = self._extract_sensitivity_features(self.S_baseline_fr)

        # Legacy placeholders (not used in fast evaluation path)
        self.current_vcm_plant = None
        self.current_pzt_plant = None

    # -----------------------------------------------------------------------
    # V1 helper kept for reference (not used in V2 _select_plant_case)
    # -----------------------------------------------------------------------

    def _peak_refs(self, peaks, low_3, high_3):
        """V1 anchor helper - superseded in V2 by _find_channel_gaps."""
        f_low, f_high = float(low_3[0]), float(high_3[0])
        refs = [p['freq'] for p in peaks if f_low <= p['freq'] <= f_high]
        fallback = float(np.sqrt(low_3[0] * high_3[0]))
        while len(refs) < self.notches_per_channel:
            refs.append(fallback)
        return refs[:self.notches_per_channel]

    # -----------------------------------------------------------------------
    # V2 - Sensitivity peak finder (raw, no padding)
    # -----------------------------------------------------------------------

    def _find_sensitivity_peaks_raw(self, sensitivity_fr, threshold_db=None):
        """
        Find local maxima in the sensitivity magnitude above threshold_db.

        Returns a list of dicts sorted by magnitude (descending):
            {'freq': Hz, 'mag': dB, 'idx': frequency-grid index}

        Unlike _find_peaks (plant FR), this does not pad the result to a fixed
        length - callers handle the empty-list case themselves.
        """
        if threshold_db is None:
            threshold_db = self.S_gap_threshold

        mag_db = 20 * np.log10(np.abs(sensitivity_fr) + 1e-12)
        peaks = []
        for i in range(1, len(mag_db) - 1):
            if (mag_db[i] > mag_db[i - 1]
                    and mag_db[i] > mag_db[i + 1]
                    and mag_db[i] > threshold_db):
                peaks.append({'freq': self.freq_range[i], 'mag': mag_db[i], 'idx': i})
        peaks.sort(key=lambda x: x['mag'], reverse=True)
        return peaks

    # -----------------------------------------------------------------------
    # V2 - Channel-specific gap attribution
    # -----------------------------------------------------------------------

    def _find_channel_gaps(self):
        """
        Attribute each S_baseline peak to the VCM or PZT notch channel.

        Decision rule (priority order):
          1. Peak only within VCM freq range -> assign to VCM
          2. Peak only within PZT freq range -> assign to PZT
          3. Peak within both ranges (overlap 10-22 kHz) -> use |L| dominance
          4. Peak outside both ranges (e.g. <5 kHz) -> skip (can't be notched)

        This prevents out-of-range peaks (e.g. 4376 Hz below VCM min 5000 Hz)
        from being wrongly assigned to a channel.

        Returns:
            vcm_gap_freqs : list of length notches_per_channel
            pzt_gap_freqs : list of length notches_per_channel
        """
        s_peaks = self._find_sensitivity_peaks_raw(self.S_baseline_fr)

        vcm_f_low  = float(self.param_low[0])
        vcm_f_high = float(self.param_high[0])
        pzt_start  = self.notches_per_channel * 3
        pzt_f_low  = float(self.param_low[pzt_start])
        pzt_f_high = float(self.param_high[pzt_start])

        # Partial loop gains at slow rate (no new notch)
        L_vcm = self.base_vcm_fr * self.Cd_vcm_fr
        L_pzt = self.base_pzt_fr * self.Cd_pzt_fr

        vcm_freqs, pzt_freqs = [], []
        for p in s_peaks:
            f   = p['freq']
            idx = p['idx']
            in_vcm = vcm_f_low <= f <= vcm_f_high
            in_pzt = pzt_f_low <= f <= pzt_f_high

            if in_vcm and not in_pzt:
                vcm_freqs.append(f)
            elif in_pzt and not in_vcm:
                pzt_freqs.append(f)
            elif in_vcm and in_pzt:
                # Overlap region: use loop-gain dominance
                if np.abs(L_vcm[idx]) >= np.abs(L_pzt[idx]):
                    vcm_freqs.append(f)
                else:
                    pzt_freqs.append(f)
            # else: outside both ranges -> skip

        # Fall back to geometric-mean of the allowed frequency range if not
        # enough real gap peaks found for this channel.
        vcm_fallback = float(np.sqrt(vcm_f_low * vcm_f_high))
        pzt_fallback = float(np.sqrt(pzt_f_low * pzt_f_high))

        while len(vcm_freqs) < self.notches_per_channel:
            vcm_freqs.append(vcm_fallback)
        while len(pzt_freqs) < self.notches_per_channel:
            pzt_freqs.append(pzt_fallback)

        return vcm_freqs[:self.notches_per_channel], pzt_freqs[:self.notches_per_channel]

    # -----------------------------------------------------------------------
    # Plant randomisation (unchanged from V1)
    # -----------------------------------------------------------------------

    def _randomize_plant(self):
        case_idx = np.random.randint(0, len(self.plant_cases))
        self._select_plant_case(self.plant_cases[case_idx][0])

    # -----------------------------------------------------------------------
    # Reset / step (unchanged from V1 except plant_features dimension)
    # -----------------------------------------------------------------------

    def reset(self):
        """Reset environment."""
        self._randomize_plant()

        if (self.initial_params_by_case is not None
                and self.current_case in self.initial_params_by_case):
            base_params = self.initial_params_by_case[self.current_case]
            base_norm = self._normalize_notch_params(base_params)
            noise = np.random.uniform(-self.init_noise, self.init_noise, self.param_dim)
            start_norm = np.clip(base_norm + noise, 0.0, 1.0)
            self.current_notch_params = self._denormalize_params(start_norm)
        else:
            self.current_notch_params = self._midpoint_params()

        performance = self._evaluate_current_system()
        self.current_performance = np.array([performance['sensitivity_peak']])

        self.current_system_features = self._extract_current_system_features(
            self.current_vcm_fr_with_notch,
            self.current_pzt_fr_with_notch,
            self.current_sensitivity_fr,
        )

        # plant_features is already set by _select_plant_case (15-dim S_baseline)
        state = np.concatenate([
            self.plant_features,
            self.current_system_features,
            self._normalize_notch_params(self.current_notch_params),
            self._normalize_performance(self.current_performance),
        ])
        return state

    def step(self, action):
        """Execute one iterative delta-move in normalised parameter space."""
        action = np.clip(action, self.action_space['low'], self.action_space['high'])
        prev_sp = float(self.current_performance[0])
        current_norm = self._normalize_notch_params(self.current_notch_params)
        new_norm = np.clip(current_norm + action * self.delta_max, 0.0, 1.0)
        self.current_notch_params = self._denormalize_params(new_norm)

        performance = self._evaluate_current_system()

        reward = self._calculate_reward(performance)
        new_sp = float(performance['sensitivity_peak'])
        improvement_bonus = max(0.0, prev_sp - new_sp) * 2.0 / self.reward_scale
        reward = float(np.clip(reward + improvement_bonus, -100.0, 10.0))

        self.current_performance = np.array([performance['sensitivity_peak']])
        self.current_system_features = self._extract_current_system_features(
            self.current_vcm_fr_with_notch,
            self.current_pzt_fr_with_notch,
            self.current_sensitivity_fr,
        )

        next_state = np.concatenate([
            self.plant_features,
            self.current_system_features,
            self._normalize_notch_params(self.current_notch_params),
            self._normalize_performance(self.current_performance),
        ])

        done = self._is_done(performance)
        info = {
            'performance': performance,
            'notch_params': self.current_notch_params,
            'case': self.current_case,
        }
        return next_state, reward, done, info

    # -----------------------------------------------------------------------
    # System evaluation (unchanged from V1)
    # -----------------------------------------------------------------------

    def _evaluate_current_system(self):
        if not hasattr(self, 'current_fast_chain') or self.current_fast_chain is None:
            raise RuntimeError("Plant must be randomized before evaluation.")

        vcm_params = self._channel_params(self.current_notch_params, 'vcm')
        pzt_params = self._channel_params(self.current_notch_params, 'pzt')
        sys_pd_vcm = self._resample_ss(
            self._series_ss(self.current_fast_chain[0], self._cascade_notch_ss(vcm_params)))
        sys_pd_pzt = self._resample_ss(
            self._series_ss(self.current_fast_chain[1], self._cascade_notch_ss(pzt_params)))

        Fr_Pd_vcm = self._freqresp_ss(sys_pd_vcm) * self.current_gain_scale
        Fr_Pd_pzt = self._freqresp_ss(sys_pd_pzt) * self.current_gain_scale

        L_vcm = Fr_Pd_vcm * self.Cd_vcm_fr
        L_pzt = Fr_Pd_pzt * self.Cd_pzt_fr
        L_total = L_vcm + L_pzt
        S = 1.0 / (1.0 + L_total)

        self.current_vcm_fr_with_notch = Fr_Pd_vcm
        self.current_pzt_fr_with_notch = Fr_Pd_pzt
        self.current_sensitivity_fr = S

        return self._calculate_performance_metrics(L_total, S)

    def _channel_params(self, params, channel):
        params = np.asarray(params, dtype=np.float64)
        if channel == 'vcm':
            start = 0
        elif channel == 'pzt':
            start = self.notches_per_channel * 3
        else:
            raise ValueError(f"Unknown channel: {channel}")
        return params[start:start + self.notches_per_channel * 3].reshape(self.notches_per_channel, 3)

    def _cascade_notch_ss(self, notch_params):
        cascade = None
        for f0, bw, depth_db in notch_params:
            notch = self._notch_ss(f0, bw, depth_db)
            cascade = self._series_ss(cascade, notch)
        return cascade

    # -----------------------------------------------------------------------
    # Feature extraction (unchanged from V1; _extract_plant_features_from_fr
    # is kept for compatibility but is NOT called in V2 _select_plant_case)
    # -----------------------------------------------------------------------

    def _extract_plant_features_from_fr(self, vcm_fr, pzt_fr):
        """V1 static plant feature extractor (30-dim). Kept for compatibility."""
        vcm_peaks = self._find_peaks(vcm_fr)
        pzt_peaks = self._find_peaks(pzt_fr)
        features = []
        for p in vcm_peaks:
            features.extend([p['freq'] / 50000.0, p['mag'] / 60.0, p['phase'] / 180.0])
        for p in pzt_peaks:
            features.extend([p['freq'] / 50000.0, p['mag'] / 60.0, p['phase'] / 180.0])
        return np.array(features, dtype=np.float32)

    def _extract_sensitivity_features(self, sensitivity_fr):
        """
        15-dim feature vector from a sensitivity FR: top-5 peaks x (freq, mag, phase).
        Used in V2 both for S_baseline (plant_features) and current S (system_features).
        """
        mag_db = 20 * np.log10(np.abs(sensitivity_fr) + 1e-12)
        peaks = []
        for i in range(1, len(mag_db) - 1):
            if mag_db[i] > mag_db[i - 1] and mag_db[i] > mag_db[i + 1] and mag_db[i] > -40:
                peaks.append({
                    'freq': self.freq_range[i],
                    'mag': mag_db[i],
                    'phase': np.angle(sensitivity_fr[i]) * 180 / np.pi,
                })
        peaks.sort(key=lambda x: x['mag'], reverse=True)
        peaks = peaks[:5]
        while len(peaks) < 5:
            peaks.append({'freq': 0, 'mag': -60, 'phase': 0})
        features = []
        for p in peaks:
            features.extend([
                p['freq'] / 50000.0,
                (p['mag'] + 40) / 40.0,
                p['phase'] / 180.0,
            ])
        return np.array(features, dtype=np.float32)

    def _extract_current_system_features(self, vcm_fr_with_notch, pzt_fr_with_notch, sensitivity_fr):
        return self._extract_sensitivity_features(sensitivity_fr)

    # -----------------------------------------------------------------------
    # Plant peak finder (unchanged from V1; still used by _peak_refs)
    # -----------------------------------------------------------------------

    def _find_peaks(self, fr):
        """Find top-5 peaks in a plant FR (threshold -20 dB)."""
        mag = 20 * np.log10(np.abs(fr) + 1e-12)
        phase = np.angle(fr) * 180 / np.pi
        peaks = []
        for i in range(1, len(mag) - 1):
            if mag[i] > mag[i - 1] and mag[i] > mag[i + 1] and mag[i] > -20:
                peaks.append({'freq': self.freq_range[i], 'mag': mag[i], 'phase': phase[i]})
        peaks.sort(key=lambda x: x['mag'], reverse=True)
        peaks = peaks[:5]
        while len(peaks) < 5:
            peaks.append({'freq': 0, 'mag': 0, 'phase': 0})
        return peaks

    # -----------------------------------------------------------------------
    # Normalisation helpers (unchanged from V1)
    # -----------------------------------------------------------------------

    def _normalize_notch_params(self, params):
        norm = np.zeros_like(params)
        for i in range(len(params)):
            low = self.param_low[i]
            high = self.param_high[i]
            val = params[i]
            if i % 3 in (0, 1):
                log_low = np.log10(low)
                log_high = np.log10(high)
                n = (np.log10(val) - log_low) / (log_high - log_low)
                norm[i] = n
            else:
                n = (val - low) / (high - low)
                norm[i] = n
        return np.clip(norm, 0.0, 1.0)

    def _denormalize_params(self, norm):
        params = np.zeros(self.param_dim, dtype=np.float64)
        for i in range(self.param_dim):
            low = float(self.param_low[i])
            high = float(self.param_high[i])
            n = float(np.clip(norm[i], 0.0, 1.0))
            if i % 3 in (0, 1):
                log_low = np.log10(low)
                log_high = np.log10(high)
                params[i] = 10.0 ** (log_low + n * (log_high - log_low))
            else:
                params[i] = low + n * (high - low)
        return params

    def _normalize_performance(self, perf):
        norm = (perf[0] + 20.0) / 40.0
        return np.array([np.clip(norm, 0.0, 1.0)])

    # -----------------------------------------------------------------------
    # Performance metrics (unchanged from V1)
    # -----------------------------------------------------------------------

    def _calculate_performance_metrics(self, Fr_L, Fr_S):
        Fr_L = np.squeeze(Fr_L)
        Fr_S = np.squeeze(Fr_S)
        if Fr_L.ndim > 1: Fr_L = Fr_L[0]
        if Fr_S.ndim > 1: Fr_S = Fr_S[0]
        pm = self._calculate_phase_margin(Fr_L)
        gm = self._calculate_gain_margin(Fr_L)
        sp = np.max(20 * np.log10(np.abs(Fr_S) + 1e-12))
        te = np.mean(20 * np.log10(np.abs(Fr_S) + 1e-12))
        return_difference_min = float(np.min(np.abs(1.0 + Fr_L)))
        finite = np.all(np.isfinite(Fr_L)) and np.all(np.isfinite(Fr_S))
        stable = 1.0 if finite and sp < 12.0 and return_difference_min > 0.15 else 0.0
        return {
            'phase_margin': float(pm),
            'gain_margin': float(gm),
            'sensitivity_peak': float(sp),
            'stability': float(stable),
            'tracking_error': float(te),
            'return_difference_min': float(return_difference_min),
        }

    def _calculate_phase_margin(self, Fr_L):
        mag = np.abs(Fr_L)
        phase = np.unwrap(np.angle(Fr_L)) * 180 / np.pi
        phase_margins = []
        for i in range(len(mag) - 1):
            if (mag[i] - 1.0) * (mag[i + 1] - 1.0) <= 0 and mag[i] != mag[i + 1]:
                frac = (1.0 - mag[i]) / (mag[i + 1] - mag[i])
                p = phase[i] + frac * (phase[i + 1] - phase[i])
                pm = 180.0 + p
                pm = ((pm + 180.0) % 360.0) - 180.0
                phase_margins.append(pm)
        if phase_margins:
            return float(min(phase_margins))
        return float('inf')

    def _calculate_gain_margin(self, Fr_L):
        mag = np.abs(Fr_L)
        phase = np.unwrap(np.angle(Fr_L)) * 180 / np.pi
        gain_margins = []
        for i in range(len(phase) - 1):
            if (phase[i] + 180.0) * (phase[i + 1] + 180.0) <= 0 and phase[i] != phase[i + 1]:
                frac = (phase[i] - (-180)) / (phase[i] - phase[i + 1] + 1e-12)
                m = mag[i] + frac * (mag[i + 1] - mag[i])
                gain_margins.append(-20 * np.log10(m + 1e-12))
        if gain_margins:
            return float(min(gain_margins))
        return float('inf')

    def _calculate_reward(self, performance):
        reward = -self._objective_from_performance(performance, self.current_notch_params) / self.reward_scale
        return float(np.clip(reward, -100.0, 10.0))

    # -----------------------------------------------------------------------
    # V2 - Objective function (relative improvement + waterbed guard)
    # -----------------------------------------------------------------------

    def _objective_from_performance(self, performance, notch_params):
        """
        V2 objective - lower is better.

        Three components:
          1. Waterbed guard  : global S_peak must not exceed
                               S_baseline_peak + S_waterbed_margin (default +0.5 dB).
                               Heavy quadratic penalty if violated.

          2. Gap improvement : at each identified gap frequency, S must decrease
                               by at least S_gap_improvement_target (default 1.0 dB)
                               vs S_baseline.  Quadratic penalty on shortfall.

          3. Stability / cost: GM, return-difference-min, and notch bandwidth/depth
                               regularisation (same as V1).
        """
        sp = float(performance['sensitivity_peak'])
        gm = float(performance['gain_margin'])
        rdm = float(performance.get('return_difference_min', 0.0))

        # --- 1. Waterbed guard (relative to this plant's baseline) ---
        # S_baseline_peak may be 0.0 until _select_plant_case has run once;
        # guard against that edge case.
        baseline_peak = self.S_baseline_peak if self.S_baseline_peak > 0.0 else sp
        waterbed_limit = baseline_peak + self.S_waterbed_margin
        # Cap excess at 5 dB so waterbed_loss stays bounded (avoids loss explosion
        # when a misplaced notch causes a large sensitivity spike elsewhere).
        waterbed_excess = min(max(0.0, sp - waterbed_limit), 5.0)
        waterbed_loss = waterbed_excess ** 2 * 20.0

        # --- 2. Gap-frequency improvement ---
        # Design intent: reward actual reduction at gap frequencies, not just
        # penalise shortfall.  Using linear improvement reward + quadratic
        # shortfall penalty gives the agent a gradient even when far from target.
        gap_loss = 0.0
        if (self.S_baseline_fr is not None
                and self.current_sensitivity_fr is not None
                and len(self.current_gap_freqs) > 0):
            for f_gap in self.current_gap_freqs:
                idx = int(np.argmin(np.abs(self.freq_range - f_gap)))
                s_now_db = 20 * np.log10(
                    np.abs(self.current_sensitivity_fr[idx]) + 1e-12)
                s_base_db = 20 * np.log10(
                    np.abs(self.S_baseline_fr[idx]) + 1e-12)
                improvement = s_base_db - s_now_db  # positive = S went down
                # Quadratic penalty when improvement < target; clipped to 3x target
                # so a large waterbed spike doesn't explode the loss.
                shortfall = min(max(0.0, self.S_gap_improvement_target - improvement),
                                3.0 * self.S_gap_improvement_target)
                gap_loss += shortfall ** 2 * 5.0
                # Linear bonus when notch actually helps (up to 2x target).
                # This creates a gradient away from trivial (depth~0) notches.
                gap_loss -= min(max(improvement, 0.0),
                                2.0 * self.S_gap_improvement_target) * 3.0

        # --- 3. Stability constraints ---
        # PM is NOT used here: for dual-actuator multirate systems, phase unwrapping
        # produces unreliable PM values when secondary gain crossovers exist.
        # Min|1+L| (rdm) is the numerically reliable stability indicator.
        gm_min = float(self.targets.get('gain_margin', 6.0))
        stability_loss = 0.5 * max(0.0, gm_min - gm) ** 2
        stability_loss += 40.0 * max(0.0, 0.30 - rdm) ** 2
        if performance['stability'] < 0.5:
            stability_loss += 20.0

        # --- 4. Notch cost regularisation ---
        # Depth regularisation is removed: penalising depth conflicted directly
        # with the gap-improvement goal and drove the agent to trivial notches.
        # Only bandwidth is regularised (wide notches hurt adjacent frequencies).
        notch_params = np.asarray(notch_params, dtype=np.float64).reshape(-1, 3)
        total_bw = float(np.sum(notch_params[:, 1]))
        notch_cost = 0.15 * (total_bw / (5000.0 * len(notch_params)))

        # --- 5. Frequency alignment ---
        # Direct gradient toward gap frequencies: penalise notch f0 being far
        # from its assigned gap target (weighted by depth so trivial notches
        # don't get large penalties that encourage staying at depth=-1 dB).
        alignment_loss = 0.0
        for i in range(len(notch_params)):
            if i < len(self.current_gap_freqs) and self.current_gap_freqs[i] > 0:
                f0        = float(notch_params[i, 0])
                depth_db  = float(notch_params[i, 2])        # negative value
                f_gap     = float(self.current_gap_freqs[i])
                log_dist  = abs(np.log10(max(f0, 1.0) / f_gap))  # 0=perfect, 1=1 octave
                depth_w   = min(abs(depth_db), 30.0) / 30.0      # [0,1], scales with depth
                alignment_loss += log_dist * depth_w * 3.0

        return float(waterbed_loss + gap_loss + stability_loss + notch_cost + alignment_loss)

    # -----------------------------------------------------------------------
    # V2 - Done condition (relative to baseline)
    # -----------------------------------------------------------------------

    def _is_done(self, performance):
        """
        Episode ends successfully when:
          - Stability constraints satisfied (GM, return-difference-min).
          - Global S_peak has not risen above baseline + waterbed_margin.
          - Mean improvement at gap frequencies >= S_gap_improvement_target.
        """
        gm = performance['gain_margin']
        gm_ok = (not np.isfinite(gm)) or gm >= self.targets['gain_margin']
        stable = performance['stability'] > 0.8
        rdm_ok = performance.get('return_difference_min', 0.0) > 0.25

        # Waterbed: global peak must not worsen beyond tolerance
        sp = performance['sensitivity_peak']
        waterbed_ok = sp <= (self.S_baseline_peak + self.S_waterbed_margin)

        # Gap improvement: average reduction at gap frequencies
        gap_ok = False
        if (self.S_baseline_fr is not None
                and self.current_sensitivity_fr is not None
                and len(self.current_gap_freqs) > 0):
            improvements = []
            for f_gap in self.current_gap_freqs:
                idx = int(np.argmin(np.abs(self.freq_range - f_gap)))
                s_now = 20 * np.log10(np.abs(self.current_sensitivity_fr[idx]) + 1e-12)
                s_base = 20 * np.log10(np.abs(self.S_baseline_fr[idx]) + 1e-12)
                improvements.append(s_base - s_now)
            gap_ok = float(np.mean(improvements)) >= self.S_gap_improvement_target

        return gm_ok and stable and rdm_ok and waterbed_ok and gap_ok


# ===========================================================================
# PPO Policy Network (unchanged from V1)
# ===========================================================================

class PPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.policy_std = nn.Linear(hidden_dim, action_dim)
        self.value_net = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m == self.policy_mean:
                    nn.init.orthogonal_(m.weight, gain=0.01)
                    nn.init.constant_(m.bias, 0)
                elif m == self.policy_std:
                    nn.init.orthogonal_(m.weight, gain=0.01)
                    nn.init.constant_(m.bias, -1.0)
                elif m == self.value_net:
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    nn.init.constant_(m.bias, 0)
                else:
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0)

    def forward(self, state, action_low=None, action_high=None):
        shared_features = self.shared_net(state)
        mean_raw = self.policy_mean(shared_features)
        mean_tanh = torch.tanh(mean_raw)
        if action_low is not None and action_high is not None:
            if isinstance(action_low, np.ndarray):
                action_low = torch.tensor(action_low, dtype=torch.float32, device=mean_tanh.device)
                action_high = torch.tensor(action_high, dtype=torch.float32, device=mean_tanh.device)
            action_mid = (action_low + action_high) / 2.0
            action_range = (action_high - action_low) / 2.0
            mean = action_mid + mean_tanh * action_range
        else:
            mean = mean_tanh
        log_std = torch.clamp(self.policy_std(shared_features), -20.0, 2.0)
        std = torch.exp(log_std)
        if action_low is not None and action_high is not None:
            if isinstance(action_low, np.ndarray):
                action_low = torch.tensor(action_low, dtype=torch.float32, device=std.device)
                action_high = torch.tensor(action_high, dtype=torch.float32, device=std.device)
            std = std * ((action_high - action_low) / 2.0)
        value = self.value_net(shared_features)
        return mean, std, value

    def get_action(self, state, action_low=None, action_high=None, deterministic=False):
        mean, std, value = self.forward(state, action_low, action_high)
        if deterministic:
            return mean, value
        dist = Normal(mean, std)
        action = dist.sample()
        if action_low is not None and action_high is not None:
            if isinstance(action_low, np.ndarray):
                action_low = torch.tensor(action_low, dtype=torch.float32, device=action.device)
                action_high = torch.tensor(action_high, dtype=torch.float32, device=action.device)
            action = torch.max(torch.min(action, action_high), action_low)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value


# ===========================================================================
# PPO Agent (unchanged from V1)
# ===========================================================================

class PPOAgent:
    def __init__(self, state_dim, action_dim, cfg):
        self.config = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPOPolicy(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.learning_rate)
        self.clip_ratio = cfg.clip_ratio
        self.value_loss_coef = cfg.value_loss_coef
        self.entropy_coef = cfg.entropy_coef
        self.max_grad_norm = cfg.max_grad_norm
        self.states = []; self.actions = []; self.rewards = []
        self.values = []; self.log_probs = []; self.dones = []

    def get_action(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_low = getattr(self, 'action_low', None)
        action_high = getattr(self, 'action_high', None)
        with torch.no_grad():
            if deterministic:
                action, value = self.policy.get_action(state_tensor, action_low, action_high, deterministic=True)
                act_np = action.cpu().numpy()[0]
                if action_low is not None:
                    act_np = np.clip(act_np, np.array(action_low), np.array(action_high))
                return act_np, value.cpu().numpy()[0]
            else:
                action, log_prob, value = self.policy.get_action(state_tensor, action_low, action_high)
                act_np = action.cpu().numpy()[0]
                if action_low is not None:
                    act_np = np.clip(act_np, np.array(action_low), np.array(action_high))
                return act_np, log_prob.cpu().numpy()[0], value.cpu().numpy()[0]

    def store_transition(self, state, action, reward, value, log_prob, done):
        self.states.append(state); self.actions.append(action)
        self.rewards.append(reward); self.values.append(value)
        self.log_probs.append(log_prob); self.dones.append(done)

    def update(self):
        if len(self.states) < self.config.batch_size:
            return
        states = torch.as_tensor(np.asarray(self.states, dtype=np.float32), device=self.device)
        actions = torch.as_tensor(np.asarray(self.actions, dtype=np.float32), device=self.device)
        old_log_probs = torch.as_tensor(np.asarray(self.log_probs, dtype=np.float32), device=self.device)
        old_values = torch.as_tensor(np.asarray(self.values, dtype=np.float32), device=self.device)
        rewards = self._compute_discounted_rewards()
        advantages = rewards - old_values.squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        action_low = getattr(self, 'action_low', None)
        action_high = getattr(self, 'action_high', None)
        for _ in range(self.config.ppo_epochs):
            mean, std, values = self.policy(states, action_low, action_high)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values.squeeze(), rewards)
            entropy_loss = -dist.entropy().mean()
            total_loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
        self.clear_buffer()

    def _compute_discounted_rewards(self):
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        discounted = np.zeros_like(rewards)
        running = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running = 0
            running = rewards[t] + self.config.gamma * running
            discounted[t] = running
        return torch.FloatTensor(discounted).to(self.device)

    def clear_buffer(self):
        self.states.clear(); self.actions.clear(); self.rewards.clear()
        self.values.clear(); self.log_probs.clear(); self.dones.clear()

    def save_model(self, filepath):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# ===========================================================================
# Stand-alone helpers (format_notch_params, evaluate_notch_params,
# build_state_for_case, optimize_action_for_case - unchanged from V1)
# ===========================================================================

def format_notch_params(params, notches_per_channel=2):
    params = np.asarray(params, dtype=np.float64)
    vcm = params[:notches_per_channel * 3].reshape(notches_per_channel, 3)
    pzt = params[notches_per_channel * 3:].reshape(notches_per_channel, 3)
    parts = []
    for idx, (f0, bw, depth) in enumerate(vcm, start=1):
        parts.append(f"VCM{idx}(f0={f0:.2f} Hz, bw={bw:.2f} Hz, depth={depth:.2f} dB)")
    for idx, (f0, bw, depth) in enumerate(pzt, start=1):
        parts.append(f"PZT{idx}(f0={f0:.2f} Hz, bw={bw:.2f} Hz, depth={depth:.2f} dB)")
    return ", ".join(parts)


def evaluate_notch_params(env, notch_params, plant_case=None):
    if plant_case is not None:
        env._select_plant_case(plant_case)
    elif env.current_case is None:
        env._randomize_plant()
    env.current_notch_params = np.array(notch_params, dtype=np.float64)
    performance = env._evaluate_current_system()
    reward = env._calculate_reward(performance)
    return performance, reward


def build_state_for_case(env, case_name):
    env._select_plant_case(case_name)
    env.current_notch_params = env._midpoint_params()
    performance = env._evaluate_current_system()
    env.current_performance = np.array([performance['sensitivity_peak']])
    env.current_system_features = env._extract_current_system_features(
        env.current_vcm_fr_with_notch,
        env.current_pzt_fr_with_notch,
        env.current_sensitivity_fr,
    )
    return np.concatenate([
        env.plant_features,
        env.current_system_features,
        env._normalize_notch_params(env.current_notch_params),
        env._normalize_performance(env.current_performance),
    ]).astype(np.float32)


def optimize_action_for_case(env, case_name, maxiter=8, popsize=5, seed=1):
    build_state_for_case(env, case_name)

    def objective(action):
        env._select_plant_case(case_name)
        params = env._action_to_params(np.asarray(action, dtype=np.float64))
        performance, _ = evaluate_notch_params(env, params, case_name)
        return env._objective_from_performance(performance, params)

    result = differential_evolution(
        objective,
        [(-1.0, 1.0)] * env.action_space['shape'][0],
        maxiter=maxiter, popsize=popsize, seed=seed, polish=True, tol=1e-3,
    )
    action = np.clip(np.asarray(result.x, dtype=np.float32), -1.0, 1.0)
    env._select_plant_case(case_name)
    params = env._action_to_params(action)
    performance, reward = evaluate_notch_params(env, params, case_name)
    actual_objective = env._objective_from_performance(performance, params)
    return action, params, performance, reward, actual_objective, result


# ===========================================================================
# Training / optimisation entry points (V2 config substituted)
# ===========================================================================

def train_simple_notch_designer(pretrain_model=None, max_episodes=None, max_steps=None, init_params=None):
    print("=== Training HDD Notch Filter Designer V2 (gap-driven, relative objective) ===")

    env_config = get_simple_config_v2()   # V2: use gap-analysis config
    env = SimpleHDDNotchDesignEnv(env_config)
    if init_params is not None:
        env.load_initial_params(init_params)

    train_config = config.TrainingConfig()
    if max_episodes is not None:
        train_config.max_episodes = int(max_episodes)
    if max_steps is not None:
        train_config.max_steps_per_episode = int(max_steps)

    state_dim = env.observation_space['shape'][0]
    action_dim = env.action_space['shape'][0]
    agent = PPOAgent(state_dim, action_dim, train_config)
    agent.action_low = env.action_space['low']
    agent.action_high = env.action_space['high']
    if pretrain_model is not None:
        agent.load_model(pretrain_model)
        print(f"Loaded pretrained model: {os.path.abspath(pretrain_model)}")

    print(f"  State dim: {state_dim}  |  Action dim: {action_dim}")
    print(f"  S_gap_improvement_target : {env.S_gap_improvement_target} dB")
    print(f"  S_waterbed_margin        : {env.S_waterbed_margin} dB")
    print(f"  S_gap_threshold          : {env.S_gap_threshold} dB")

    episode_rewards = []
    avg_rewards = []      # rolling average over log_interval episodes
    best_rewards = []     # running best up to each episode
    best_reward = float('-inf')
    best_eval_reward = float('-inf')   # best average across ALL 9 cases
    eval_interval = 100                # evaluate all cases every N episodes
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_path     = os.path.join(models_dir, f"v2_notch_best_{timestamp}.pth")
    best_mc_path  = os.path.join(models_dir, f"v2_notch_best_allcases_{timestamp}.pth")

    def _eval_all_cases():
        """Run current policy on all 9 cases, return mean episode reward."""
        rewards = []
        for pc in env.plant_cases:
            case_name = pc[0]
            env._select_plant_case(case_name)
            if (env.initial_params_by_case is not None
                    and case_name in env.initial_params_by_case):
                env.current_notch_params = env.initial_params_by_case[case_name].copy()
            else:
                env.current_notch_params = env._midpoint_params()
            performance = env._evaluate_current_system()
            env.current_performance = np.array([performance['sensitivity_peak']])
            env.current_system_features = env._extract_current_system_features(
                env.current_vcm_fr_with_notch,
                env.current_pzt_fr_with_notch,
                env.current_sensitivity_fr,
            )
            s = np.concatenate([
                env.plant_features,
                env.current_system_features,
                env._normalize_notch_params(env.current_notch_params),
                env._normalize_performance(env.current_performance),
            ])
            ep_r = 0.0
            for _ in range(train_config.max_steps_per_episode):
                with __import__('torch').no_grad():
                    a, _, _ = agent.get_action(s)
                s, r, done, _ = env.step(a)
                ep_r += r
                if done:
                    break
            rewards.append(ep_r)
        return float(np.mean(rewards))

    for episode in range(train_config.max_episodes):
        state = env.reset()
        episode_reward = 0
        for step in range(train_config.max_steps_per_episode):
            action, log_prob, value = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, value, log_prob, done)
            state = next_state
            episode_reward += reward
            if done:
                break
        agent.update()
        episode_rewards.append(episode_reward)
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model(best_path)
        best_rewards.append(best_reward)
        # Every eval_interval episodes: evaluate on ALL cases and save separate best
        if episode % eval_interval == 0:
            eval_r = _eval_all_cases()
            if eval_r > best_eval_reward:
                best_eval_reward = eval_r
                agent.save_model(best_mc_path)
        if episode % train_config.log_interval == 0:
            avg = np.mean(episode_rewards[-train_config.log_interval:])
            avg_rewards.append((episode, avg))
            print(f"Episode {episode:5d} | avg_reward={avg:.3f} | best={best_reward:.3f} | best_allcases={best_eval_reward:.3f}")

    rewards_array = np.array(episode_rewards, dtype=np.float32)
    np.save(os.path.join(models_dir, f"v2_rewards_{timestamp}.npy"), rewards_array)
    np.savetxt(
        os.path.join(models_dir, f"v2_rewards_{timestamp}.csv"),
        rewards_array, fmt='%.6f', delimiter=',', header='episode_reward', comments='',
    )

    # Plot: episode reward, rolling avg, and running best on the same figure
    avg_ep, avg_vals = zip(*avg_rewards) if avg_rewards else ([], [])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(episode_rewards, color='steelblue', alpha=0.3, linewidth=0.6, label='Episode Reward')
    ax.plot(avg_ep, avg_vals, color='steelblue', linewidth=2.0,
            label=f'Avg Reward (window={train_config.log_interval})')
    ax.plot(best_rewards, color='tomato', linewidth=1.5, linestyle='--', label='Running Best')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('V2 Notch Designer Training Rewards')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, f"v2_rewards_{timestamp}.png"), dpi=120)
    plt.close()
    final_path = os.path.join(models_dir, f"v2_final_{timestamp}.pth")
    agent.save_model(final_path)
    print(f"Saved final model: {os.path.abspath(final_path)}")


def optimize_notch_designer(plant_case='c2', maxiter=25, popsize=8, seed=1, workers=1):
    print("=== V2 Notch Optimization (gap-driven objective) ===")
    if workers != 1:
        print("Parallel workers disabled on Windows; using workers=1.")
        workers = 1
    env = SimpleHDDNotchDesignEnv(get_simple_config_v2())   # V2 config

    if plant_case == 'all':
        case_names = [c[0] for c in env.plant_cases]
    else:
        case_name = plant_case if str(plant_case).startswith('c') else f'c{plant_case}'
        known = {c[0] for c in env.plant_cases}
        if case_name not in known:
            raise ValueError(f"Unknown plant case: {plant_case}. Available: {sorted(known)} or 'all'.")
        case_names = [case_name]

    bounds = list(zip(env.param_low, env.param_high))

    def objective(params):
        losses = []
        for cn in case_names:
            performance, _ = evaluate_notch_params(env, params, cn)
            losses.append(env._objective_from_performance(performance, params))
        return float(np.mean(losses))

    result = differential_evolution(
        objective, bounds, maxiter=maxiter, popsize=popsize, seed=seed,
        polish=True, workers=workers,
        updating='immediate' if workers == 1 else 'deferred', tol=1e-3,
    )
    best_params = np.array(result.x, dtype=np.float64)
    print(f"Best objective: {result.fun:.4f}")
    print(f"Best params: {format_notch_params(best_params, env.notches_per_channel)}")
    for cn in case_names:
        perf, reward = evaluate_notch_params(env, best_params, cn)
        print(
            f"  {cn}: reward={reward:.4f}  S_peak={perf['sensitivity_peak']:.2f} dB  "
            f"S_baseline={env.S_baseline_peak:.2f} dB  "
            f"GM={perf['gain_margin']:.2f} dB  stable={perf['stability']:.0f}"
        )
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(models_dir, f"v2_optimized_{plant_case}_{timestamp}.npz")
    np.savez(out, notch_params=best_params, objective=float(result.fun),
             plant_case=str(plant_case), maxiter=int(maxiter),
             popsize=int(popsize), seed=int(seed))
    print(f"Saved: {os.path.abspath(out)}")
    return best_params, result


def pretrain_policy_with_optimizer(maxiter=8, popsize=5, epochs=600, seed=1):
    print("=== V2 Supervised Pretrain from Local Optimizer Baselines ===")
    env = SimpleHDDNotchDesignEnv(get_simple_config_v2())   # V2 config
    train_config = config.TrainingConfig()
    state_dim = env.observation_space['shape'][0]
    action_dim = env.action_space['shape'][0]
    agent = PPOAgent(state_dim, action_dim, train_config)
    agent.action_low = env.action_space['low']
    agent.action_high = env.action_space['high']

    states, actions, rows = [], [], []
    for idx, (case_name, _, _) in enumerate(env.plant_cases):
        action, params, performance, reward, actual_obj, result = optimize_action_for_case(
            env, case_name, maxiter=maxiter, popsize=popsize, seed=seed + idx)
        state = build_state_for_case(env, case_name)
        states.append(state); actions.append(action)
        rows.append((case_name, params, performance, reward, actual_obj))
        print(
            f"  {case_name}: obj={actual_obj:.4f}  reward={reward:.4f}  "
            f"S_peak={performance['sensitivity_peak']:.2f} dB  "
            f"S_baseline={env.S_baseline_peak:.2f} dB  "
            f"GM={performance['gain_margin']:.2f} dB"
        )

    states_t = torch.as_tensor(np.asarray(states, dtype=np.float32), device=agent.device)
    actions_t = torch.as_tensor(np.asarray(actions, dtype=np.float32), device=agent.device)
    for epoch in range(int(epochs)):
        mean, _, _ = agent.policy(states_t)
        loss = nn.MSELoss()(mean, actions_t)
        agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), agent.max_grad_norm)
        agent.optimizer.step()
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Pretrain epoch {epoch:4d}: mse={loss.item():.6f}")

    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(models_dir, f"v2_pretrained_{timestamp}.pth")
    data_path = os.path.join(models_dir, f"v2_pretrain_dataset_{timestamp}.npz")
    agent.save_model(model_path)
    np.savez(data_path,
             states=np.asarray(states, dtype=np.float32),
             actions=np.asarray(actions, dtype=np.float32),
             cases=np.asarray([r[0] for r in rows]),
             notch_params=np.asarray([r[1] for r in rows], dtype=np.float64),
             rewards=np.asarray([r[3] for r in rows], dtype=np.float64))
    print(f"Saved pretrained policy : {os.path.abspath(model_path)}")
    print(f"Saved pretrain dataset  : {os.path.abspath(data_path)}")
    return model_path


def load_model_and_predict(model_path, env=None, deterministic=True):
    """Compatibility helper used by evaluate_model.py."""
    if env is None:
        env = SimpleHDDNotchDesignEnv(get_simple_config_v2())   # V2 config
    train_config = config.TrainingConfig()
    state_dim = env.observation_space['shape'][0]
    action_dim = env.action_space['shape'][0]
    agent = PPOAgent(state_dim, action_dim, train_config)
    agent.action_low = env.action_space['low']
    agent.action_high = env.action_space['high']
    agent.load_model(model_path)
    state = env.reset()
    if deterministic:
        action, value = agent.get_action(state, deterministic=True)
    else:
        action, _, value = agent.get_action(state, deterministic=False)
    _, reward, done, info = env.step(action)
    return {
        'action': action, 'value': value, 'reward': reward,
        'done': done, 'performance': info['performance'],
        'notch_params': info['notch_params'], 'case': info['case'],
    }


# ===========================================================================
# CLI (identical to V1)
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="HDD notch filter design V2")
    parser.add_argument('--mode', choices=['train', 'optimize', 'pretrain'], default='train')
    parser.add_argument('--plant-case', default='c2')
    parser.add_argument('--maxiter', type=int, default=25)
    parser.add_argument('--popsize', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--pretrain-model', default=None)
    parser.add_argument('--max-episodes', type=int, default=None)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--pretrain-epochs', type=int, default=600)
    parser.add_argument('--init-params', default=None,
                        help="Path to optimized_notch_*.npz for DE warm-start.")
    args = parser.parse_args()

    if args.mode == 'train':
        train_simple_notch_designer(
            pretrain_model=args.pretrain_model,
            max_episodes=args.max_episodes,
            max_steps=args.max_steps,
            init_params=args.init_params,
        )
    elif args.mode == 'optimize':
        optimize_notch_designer(
            plant_case=args.plant_case,
            maxiter=args.maxiter,
            popsize=args.popsize,
            seed=args.seed,
            workers=args.workers,
        )
    else:
        pretrain_policy_with_optimizer(
            maxiter=args.maxiter,
            popsize=args.popsize,
            epochs=args.pretrain_epochs,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
