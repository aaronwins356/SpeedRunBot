"""
helpers.py - Utility functions for Minecraft RL agent.

This module provides common utility functions for:
- Configuration loading
- Random seed management
- Data processing
- File operations
"""

import numpy as np
import random
import os
import json
from typing import Dict, Optional, Any, List
from pathlib import Path


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Sets seed for:
    - NumPy random
    - Python random
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Set environment variable for any other libraries
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_config(path: str) -> Optional[Dict]:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        path: Path to configuration file
        
    Returns:
        Configuration dictionary, or None if file not found
    """
    if not os.path.exists(path):
        print(f"Config file not found: {path}")
        return None
    
    try:
        with open(path, 'r') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                # Simple YAML parsing (subset of YAML)
                return _parse_simple_yaml(f.read())
            else:
                return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return None


def _parse_simple_yaml(content: str) -> Dict:
    """
    Parse a simple YAML file (key: value pairs).
    
    This is a minimal YAML parser for basic config files.
    
    WARNING: This parser is for trusted config files only.
    For production use with untrusted input, use PyYAML with
    safe_load() to prevent code injection attacks.
    
    Args:
        content: YAML file content (from trusted source)
        
    Returns:
        Parsed dictionary
    """
    result = {}
    current_section = result
    
    for line in content.split('\n'):
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        
        # Handle key: value pairs
        if ':' in line:
            key, _, value = line.partition(':')
            key = key.strip()
            value = value.strip()
            
            # Basic validation: key should be alphanumeric with underscores
            if not key.replace('_', '').replace('-', '').isalnum():
                continue  # Skip invalid keys
            
            if value:
                # Parse value type
                current_section[key] = _parse_yaml_value(value)
            else:
                # New section (not fully supported in this minimal parser)
                current_section[key] = {}
    
    return result


def _parse_yaml_value(value: str) -> Any:
    """Parse a YAML value string into appropriate Python type."""
    # Remove quotes
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    
    # Boolean
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    
    # None
    if value.lower() in ('null', 'none', '~'):
        return None
    
    # Number
    try:
        if '.' in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    
    # String
    return value


def save_config(config: Dict, path: str) -> None:
    """
    Save configuration to a file.
    
    Args:
        config: Configuration dictionary
        path: Path to save to
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        if path.endswith('.yaml') or path.endswith('.yml'):
            # Simple YAML output
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
        else:
            json.dump(config, f, indent=2)


def normalize_observation(obs: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Normalize observation array.
    
    Args:
        obs: Observation array
        method: Normalization method ('standard', 'minmax', 'none')
        
    Returns:
        Normalized observation
    """
    if method == 'none':
        return obs
    
    if method == 'minmax':
        min_val = obs.min()
        max_val = obs.max()
        if max_val - min_val > 0:
            return (obs - min_val) / (max_val - min_val)
        return obs - min_val
    
    if method == 'standard':
        mean = obs.mean()
        std = obs.std()
        if std > 0:
            return (obs - mean) / std
        return obs - mean
    
    return obs


def compute_discounted_returns(
    rewards: List[float],
    gamma: float = 0.99
) -> np.ndarray:
    """
    Compute discounted returns from rewards.
    
    Args:
        rewards: List of rewards
        gamma: Discount factor
        
    Returns:
        Array of discounted returns
    """
    n = len(rewards)
    returns = np.zeros(n)
    
    running_return = 0.0
    for t in reversed(range(n)):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    
    return returns


def compute_gae(
    rewards: List[float],
    values: List[float],
    gamma: float = 0.99,
    lambda_: float = 0.95
) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: List of rewards
        values: List of value estimates
        gamma: Discount factor
        lambda_: GAE lambda parameter
        
    Returns:
        Array of advantages
    """
    n = len(rewards)
    advantages = np.zeros(n)
    
    last_gae = 0.0
    for t in reversed(range(n)):
        next_value = values[t + 1] if t + 1 < len(values) else 0
        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = last_gae = delta + gamma * lambda_ * last_gae
    
    return advantages


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute explained variance.
    
    Useful for evaluating value function quality.
    
    Args:
        y_pred: Predicted values
        y_true: True values
        
    Returns:
        Explained variance (1.0 = perfect, 0 = bad)
    """
    var_y = np.var(y_true)
    if var_y == 0:
        return 0.0
    return 1 - np.var(y_true - y_pred) / var_y


def smooth_values(values: List[float], window: int = 10) -> List[float]:
    """
    Apply moving average smoothing to values.
    
    Args:
        values: List of values
        window: Smoothing window size
        
    Returns:
        Smoothed values
    """
    if len(values) < window:
        return values
    
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(values[start:i + 1]))
    
    return smoothed


def create_batches(
    data: Dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool = True
) -> List[Dict[str, np.ndarray]]:
    """
    Create batches from data dictionary.
    
    Args:
        data: Dictionary with arrays of same length
        batch_size: Size of each batch
        shuffle: Whether to shuffle before batching
        
    Returns:
        List of batch dictionaries
    """
    # Get data length
    n = len(next(iter(data.values())))
    
    # Create indices
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    
    # Create batches
    batches = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_indices = indices[start:end]
        
        batch = {
            key: value[batch_indices] if isinstance(value, np.ndarray) else value
            for key, value in data.items()
        }
        batches.append(batch)
    
    return batches


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_progress_bar(
    current: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    length: int = 50,
    fill: str = "█"
) -> None:
    """
    Print a progress bar.
    
    Args:
        current: Current progress value
        total: Total value
        prefix: Prefix string
        suffix: Suffix string
        length: Bar length in characters
        fill: Fill character
    """
    percent = current / total
    filled = int(length * percent)
    bar = fill * filled + "-" * (length - filled)
    
    print(f"\r{prefix} |{bar}| {percent:.1%} {suffix}", end="", flush=True)
    
    if current == total:
        print()


def ensure_dir(path: str) -> str:
    """
    Ensure directory exists.
    
    Args:
        path: Directory path
        
    Returns:
        The path
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
