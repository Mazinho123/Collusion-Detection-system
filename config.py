"""
Configuration Module for Bidder Collusion Detection System
Provides default configurations and preset profiles for different use cases
"""

import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class DetectionConfig:
    """Main configuration class for the collusion detection system"""
    
    # Data Generation Parameters
    num_auctions: int = 100
    num_bidders: int = 50
    num_collusion_groups: int = 3
    collusion_percentage: float = 0.25
    seed: int = None  # None = random seed each time; set to fixed value for reproducibility
    
    # Feature Engineering Parameters
    feature_scaling: str = "standard"  # standard, minmax, robust
    outlier_threshold: float = 3.0
    correlation_threshold: float = 0.9
    
    # Anomaly Detection Parameters
    isolation_forest_contamination: float = 0.1
    isolation_forest_n_estimators: int = 100
    lof_n_neighbors: int = 20
    lof_contamination: float = 0.1
    ensemble_method: str = "voting"  # voting, averaging
    
    # Network Analysis Parameters
    min_coprediction_threshold: int = 3
    community_detection_resolution: float = 1.0
    network_visualization_threshold: float = 0.3
    
    # Risk Scoring Parameters
    anomaly_score_weight: float = 0.5
    network_score_weight: float = 0.3
    behavior_score_weight: float = 0.2
    risk_threshold_low: float = 0.3
    risk_threshold_medium: float = 0.6
    risk_threshold_high: float = 0.8
    
    # Explainability Parameters
    shap_sample_size: int = 100
    permutation_importance_iterations: int = 10
    top_features_to_show: int = 5
    
    # Visualization Parameters
    figure_dpi: int = 100
    figure_style: str = "seaborn-v0_8-darkgrid"
    color_palette: str = "husl"
    plot_output_format: str = "png"  # png, pdf, svg
    
    # Output and Logging Parameters
    output_directory: str = "./results"
    log_level: str = "INFO"
    save_intermediate_results: bool = True
    verbose: bool = True
    
    # Model Parameters
    random_state: int = 42
    n_jobs: int = -1  # Use all available cores
    batch_size: int = 32
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    def to_json(self, filepath: str) -> None:
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'DetectionConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def get_conservative_preset(cls) -> 'DetectionConfig':
        """Conservative configuration - high precision, lower recall"""
        config = cls()
        config.isolation_forest_contamination = 0.05
        config.lof_contamination = 0.05
        config.risk_threshold_high = 0.85
        config.anomaly_score_weight = 0.6
        config.network_score_weight = 0.25
        config.behavior_score_weight = 0.15
        return config
    
    @classmethod
    def get_balanced_preset(cls) -> 'DetectionConfig':
        """Balanced configuration - default"""
        return cls()
    
    @classmethod
    def get_aggressive_preset(cls) -> 'DetectionConfig':
        """Aggressive configuration - high recall, lower precision"""
        config = cls()
        config.isolation_forest_contamination = 0.15
        config.lof_contamination = 0.15
        config.risk_threshold_high = 0.75
        config.anomaly_score_weight = 0.4
        config.network_score_weight = 0.35
        config.behavior_score_weight = 0.25
        return config
    
    @classmethod
    def get_network_focused_preset(cls) -> 'DetectionConfig':
        """Network-focused configuration"""
        config = cls()
        config.network_score_weight = 0.5
        config.anomaly_score_weight = 0.3
        config.behavior_score_weight = 0.2
        config.min_coprediction_threshold = 2
        return config
    
    @classmethod
    def get_behavior_focused_preset(cls) -> 'DetectionConfig':
        """Behavior pattern-focused configuration"""
        config = cls()
        config.behavior_score_weight = 0.4
        config.anomaly_score_weight = 0.4
        config.network_score_weight = 0.2
        return config
    
    @classmethod
    def get_large_scale_preset(cls) -> 'DetectionConfig':
        """Configuration for large-scale datasets"""
        config = cls()
        config.num_auctions = 1000
        config.num_bidders = 500
        config.isolation_forest_contamination = 0.08
        config.lof_n_neighbors = 50
        config.n_jobs = -1
        config.batch_size = 64
        return config


# Default configuration instance
DEFAULT_CONFIG = DetectionConfig()

# Configuration presets
CONFIG_PRESETS = {
    'conservative': DetectionConfig.get_conservative_preset,
    'balanced': DetectionConfig.get_balanced_preset,
    'aggressive': DetectionConfig.get_aggressive_preset,
    'network_focused': DetectionConfig.get_network_focused_preset,
    'behavior_focused': DetectionConfig.get_behavior_focused_preset,
    'large_scale': DetectionConfig.get_large_scale_preset,
}


def load_config(preset_name: Optional[str] = None, json_filepath: Optional[str] = None) -> DetectionConfig:
    """
    Load configuration from preset or JSON file
    
    Args:
        preset_name: Name of preset configuration
        json_filepath: Path to JSON configuration file
    
    Returns:
        DetectionConfig instance
    """
    if json_filepath:
        return DetectionConfig.from_json(json_filepath)
    elif preset_name and preset_name in CONFIG_PRESETS:
        return CONFIG_PRESETS[preset_name]()
    else:
        return DEFAULT_CONFIG
