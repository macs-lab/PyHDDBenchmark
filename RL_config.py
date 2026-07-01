"""
HDD Notch Filter Auto Design Tool - Configuration File
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SystemConfig:
    """System Configuration"""
    # Sampling time
    Ts: float = 1.9841e-05
    
    # Multi-rate factor
    Mr_f: int = 2
    
    # Plant case - only handle one case
    plant_case: str = 'RT'  # Fixed use RT case


@dataclass
class TrainingConfig:
    """Training Configuration"""
    # Training parameters
    max_episodes: int = 3000
    max_steps_per_episode: int = 30
    log_interval: int = 50
    
    # PPO parameters
    learning_rate: float = 2e-5
    gamma: float = 0.99
    clip_ratio: float = 0.15
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.005
    ppo_epochs: int = 4
    batch_size: int = 64
    max_grad_norm: float = 0.5
    hidden_dim: int = 256


@dataclass
class NotchFilterConfig:
    """Notch Filter Configuration"""
    # VCM parameter ranges
    # VCM resonances within slow-rate Nyquist (~25 kHz): 5.3k, 6.1k, 6.5k, 8k, 9.6k, 14.8k, 17.4k, 21k Hz
    # Upper limit 16 kHz: VCM sensitivity peaks are in 8-12 kHz range; 16-22 kHz
    # is PZT-dominated and L_vcm is negligible there, so VCM notches have no effect.
    vcm_center_freq_range: tuple = (5000, 16000)  # Hz
    vcm_bandwidth_range: tuple = (100, 3000)       # Hz - narrowed to reduce over-broad notches
    vcm_depth_range: tuple = (-40, -1)             # dB

    # PZT parameter ranges
    # PZT resonances within slow-rate Nyquist: 14.8k, 21.5k Hz
    pzt_center_freq_range: tuple = (10000, 22000)  # Hz
    pzt_bandwidth_range: tuple = (100, 3000)        # Hz
    pzt_depth_range: tuple = (-40, -1)              # dB


@dataclass
class PerformanceTargets:
    """Performance Targets"""
    phase_margin: float = 45.0      # degrees
    gain_margin: float = 6.0        # dB
    sensitivity_peak: float = 5.0  # dB (target notch reduces sensitivity hump)
    stability: float = 1.0          # stability metric
    tracking_error: float = -20.0     # tracking error in dB (approx 10% error)


@dataclass
class RewardWeights:
    """Reward Weights"""
    phase_margin: float = 0 #1.0
    gain_margin: float = 0 #1.0
    sensitivity_peak: float = 15
    stability: float = 0 #2.0
    tracking_error: float = 0 #0.7


class ConfigManager:
    """Configuration Manager"""
    
    def __init__(self):
        self.system_config = SystemConfig()
        self.training_config = TrainingConfig()
        self.notch_filter_config = NotchFilterConfig()
        self.performance_targets = PerformanceTargets()
        self.reward_weights = RewardWeights()
    
    def get_env_config(self) -> Dict[str, Any]:
        """Get environment configuration"""
        return {
            'weights': {
                'phase_margin': self.reward_weights.phase_margin,
                'gain_margin': self.reward_weights.gain_margin,
                'sensitivity_peak': self.reward_weights.sensitivity_peak,
                'stability': self.reward_weights.stability,
                'tracking_error': self.reward_weights.tracking_error
            },
            'targets': {
                'phase_margin': self.performance_targets.phase_margin,
                'gain_margin': self.performance_targets.gain_margin,
                'sensitivity_peak': self.performance_targets.sensitivity_peak,
                'stability': self.performance_targets.stability,
                'tracking_error': self.performance_targets.tracking_error
            }
        }
    
    def get_action_space_config(self) -> Dict[str, Any]:
        """Get action space configuration"""
        return {
            'low': [
                self.notch_filter_config.vcm_center_freq_range[0],
                self.notch_filter_config.vcm_bandwidth_range[0],
                self.notch_filter_config.vcm_depth_range[0],
                self.notch_filter_config.pzt_center_freq_range[0],
                self.notch_filter_config.pzt_bandwidth_range[0],
                self.notch_filter_config.pzt_depth_range[0]
            ],
            'high': [
                self.notch_filter_config.vcm_center_freq_range[1],
                self.notch_filter_config.vcm_bandwidth_range[1],
                self.notch_filter_config.vcm_depth_range[1],
                self.notch_filter_config.pzt_center_freq_range[1],
                self.notch_filter_config.pzt_bandwidth_range[1],
                self.notch_filter_config.pzt_depth_range[1]
            ]
        }
    
    def update_performance_targets(self, **kwargs):
        """Update performance targets"""
        for key, value in kwargs.items():
            if hasattr(self.performance_targets, key):
                setattr(self.performance_targets, key, value)
    
    def update_reward_weights(self, **kwargs):
        """Update reward weights"""
        for key, value in kwargs.items():
            if hasattr(self.reward_weights, key):
                setattr(self.reward_weights, key, value)
    
    def update_training_config(self, **kwargs):
        """Update training configuration"""
        for key, value in kwargs.items():
            if hasattr(self.training_config, key):
                setattr(self.training_config, key, value)


def get_default_config() -> ConfigManager:
    """Get default configuration"""
    return ConfigManager()


def get_simple_config() -> Dict[str, Any]:
    """Get simple configuration (for simple_rl_notch_designer.py)"""
    manager = ConfigManager()
    env_config = manager.get_env_config()
    env_config['action_bounds'] = manager.get_action_space_config()
    # Delta action scale: fraction of normalized range moved per RL step.
    # With 0.12 and 30 steps, the agent can traverse the full range ~3.6x.
    env_config['delta_max'] = 0.12
    return env_config


def get_custom_config(
    phase_margin_target: float = 45.0,
    gain_margin_target: float = 6.0,
    sensitivity_peak_target: float = -6.0,
    stability_weight: float = 2.0,
    max_episodes: int = 1000
) -> ConfigManager:
    """Get custom configuration"""
    config = ConfigManager()
    
    # Update performance targets
    config.performance_targets.phase_margin = phase_margin_target
    config.performance_targets.gain_margin = gain_margin_target
    config.performance_targets.sensitivity_peak = sensitivity_peak_target
    
    # Update reward weights
    config.reward_weights.stability = stability_weight
    
    # Update training configuration
    config.training_config.max_episodes = max_episodes
    
    return config


# Preset configurations
PRESET_CONFIGS = {
    'conservative': {
        'phase_margin_target': 60.0,
        'gain_margin_target': 8.0,
        'sensitivity_peak_target': -8.0,
        'stability_weight': 3.0
    },
    'balanced': {
        'phase_margin_target': 45.0,
        'gain_margin_target': 6.0,
        'sensitivity_peak_target': -6.0,
        'stability_weight': 2.0
    },
    'aggressive': {
        'phase_margin_target': 30.0,
        'gain_margin_target': 4.0,
        'sensitivity_peak_target': -4.0,
        'stability_weight': 1.0
    }
}


def get_preset_config(preset_name: str) -> ConfigManager:
    """Get preset configuration"""
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESET_CONFIGS.keys())}")
    
    return get_custom_config(**PRESET_CONFIGS[preset_name])


if __name__ == "__main__":
    # Test configuration
    config = get_default_config()
    print("Default configuration:")
    print(f"  Phase margin target: {config.performance_targets.phase_margin} deg")
    print(f"  Gain margin target: {config.performance_targets.gain_margin}dB")
    print(f"  Sensitivity peak target: {config.performance_targets.sensitivity_peak}dB")
    print(f"  Stability weight: {config.reward_weights.stability}")
    
    # Test preset configuration
    conservative_config = get_preset_config('conservative')
    print("\nConservative configuration:")
    print(f"  Phase margin target: {conservative_config.performance_targets.phase_margin} deg")
    print(f"  Gain margin target: {conservative_config.performance_targets.gain_margin}dB")
    print(f"  Sensitivity peak target: {conservative_config.performance_targets.sensitivity_peak}dB")
    print(f"  Stability weight: {conservative_config.reward_weights.stability}")
