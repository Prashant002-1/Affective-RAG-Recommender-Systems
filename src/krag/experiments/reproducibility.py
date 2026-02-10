"""
Reproducibility Utilities for Affective-RAG Research
Ensures consistent results across experiment runs
"""

import random
import json
import hashlib
from datetime import datetime
from typing import Dict, Any
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)

    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_experiment_id() -> str:
    """
    Generate unique experiment ID based on timestamp.

    Returns:
        Unique experiment identifier (format: exp_YYYYMMDD_HHMMSS)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"exp_{timestamp}"


def get_config_hash(config: Dict[str, Any]) -> str:
    """
    Generate hash of configuration for tracking.

    Args:
        config: Configuration dictionary

    Returns:
        MD5 hash of configuration (first 8 characters)
    """
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def save_experiment_config(config: Dict[str, Any], path: str):
    """
    Save experiment configuration to file.

    Args:
        config: Configuration dictionary
        path: Output file path
    """
    config_with_meta = {
        **config,
        '_timestamp': datetime.now().isoformat(),
        '_config_hash': get_config_hash(config)
    }

    with open(path, 'w') as f:
        json.dump(config_with_meta, f, indent=2)


def load_experiment_config(path: str) -> Dict[str, Any]:
    """
    Load experiment configuration from file.

    Args:
        path: Configuration file path

    Returns:
        Configuration dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)


def verify_reproducibility(config1: Dict[str, Any], config2: Dict[str, Any]) -> bool:
    """
    Verify two configurations produce same experiment setup.

    Args:
        config1: First configuration
        config2: Second configuration

    Returns:
        True if configurations are equivalent
    """
    # Remove metadata fields for comparison
    def clean_config(config):
        return {k: v for k, v in config.items() if not k.startswith('_')}

    return get_config_hash(clean_config(config1)) == get_config_hash(clean_config(config2))


class ExperimentTracker:
    """
    Track experiment runs and results for reproducibility.

    Usage:
        tracker = ExperimentTracker("./results")
        tracker.log_run(config, results)
        tracker.get_best_run("NDCG@10")
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.runs = []

    def log_run(
        self,
        config: Dict[str, Any],
        results: Dict[str, float],
        notes: str = ""
    ):
        """Log an experiment run"""
        run = {
            'experiment_id': get_experiment_id(),
            'config_hash': get_config_hash(config),
            'config': config,
            'results': results,
            'notes': notes,
            'timestamp': datetime.now().isoformat()
        }
        self.runs.append(run)

    def get_best_run(self, metric: str) -> Dict[str, Any]:
        """Get run with best performance on given metric"""
        if not self.runs:
            return {}

        return max(self.runs, key=lambda r: r['results'].get(metric, 0))

    def save_history(self, path: str):
        """Save all runs to file"""
        with open(path, 'w') as f:
            json.dump(self.runs, f, indent=2)

    def load_history(self, path: str):
        """Load runs from file"""
        with open(path, 'r') as f:
            self.runs = json.load(f)
