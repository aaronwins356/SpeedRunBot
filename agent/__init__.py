"""
Agent module for Minecraft RL Bot.

This module provides the PyTorch neural network models and policy interfaces:
- MinecraftPolicy: Main actor-critic model
- Policy: High-level policy wrapper with exploration
- RolloutBuffer: Experience storage for training
"""

from .model import (
    ModelConfig,
    BlockEncoder,
    StateEncoder,
    MinecraftPolicy,
    create_model
)
from .policy import (
    PolicyConfig,
    Policy,
    RolloutBuffer
)

__all__ = [
    'ModelConfig',
    'BlockEncoder',
    'StateEncoder',
    'MinecraftPolicy',
    'create_model',
    'PolicyConfig',
    'Policy',
    'RolloutBuffer'
]
