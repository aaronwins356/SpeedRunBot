"""
config.py - Configuration management for Minecraft RL Bot.

This module provides utilities for:
- Loading configuration from YAML/JSON files
- Setting random seeds for reproducibility
- Managing hyperparameters
"""

import os
import json
import random
import numpy as np
import torch
from typing import Dict, Optional, Any
from pathlib import Path


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Sets seed for:
    - NumPy random
    - Python random
    - PyTorch (CPU and CUDA)
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set environment variable for any other libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Ensure deterministic operations in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
                # Use PyYAML if available, fallback to simple parser
                try:
                    import yaml
                    return yaml.safe_load(f)
                except ImportError:
                    f.seek(0)
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
    For complex YAML, install PyYAML: pip install pyyaml
    
    Args:
        content: YAML file content
        
    Returns:
        Parsed dictionary
    """
    result = {}
    
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
                continue
            
            if value:
                result[key] = _parse_yaml_value(value)
    
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
            try:
                import yaml
                yaml.dump(config, f, default_flow_style=False)
            except ImportError:
                # Simple YAML output
                for key, value in config.items():
                    f.write(f"{key}: {value}\n")
        else:
            json.dump(config, f, indent=2)


class Config:
    """
    Configuration class for managing hyperparameters.
    
    Supports loading from files and command-line overrides.
    
    Usage:
        config = Config.from_yaml('config.yaml')
        config.learning_rate = 0.001
        config.save('config_modified.yaml')
    """
    
    # Default configuration values
    DEFAULTS = {
        'num_episodes': 10000,
        'max_steps_per_episode': 10000,
        'batch_size': 64,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'learning_rate': 0.0003,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'log_interval': 10,
        'save_interval': 100,
        'eval_interval': 50,
        'checkpoint_dir': 'checkpoints',
        'use_curriculum': True,
        'continuous_actions': False,
        'seed': 42
    }
    
    def __init__(self, **kwargs):
        """Initialize config with defaults and overrides."""
        # Set defaults
        for key, value in self.DEFAULTS.items():
            setattr(self, key, value)
        
        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(self, key) or key in self.DEFAULTS:
                setattr(self, key, value)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        config_dict = load_config(path) or {}
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, path: str) -> 'Config':
        """Load configuration from JSON file."""
        config_dict = load_config(path) or {}
        return cls(**config_dict)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            key: getattr(self, key)
            for key in self.DEFAULTS.keys()
        }
    
    def save(self, path: str) -> None:
        """Save configuration to file."""
        save_config(self.to_dict(), path)
    
    def __repr__(self) -> str:
        items = ', '.join(f'{k}={v}' for k, v in self.to_dict().items())
        return f'Config({items})'
