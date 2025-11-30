"""
Environment module for Minecraft RL Bot.

This module provides environment implementations:
- MinecraftEnv: Base mock environment for training
- LemonCloudEnv: Live environment wrapper for Minecraft server connection
- DiscreteAction: Action representation for discrete action space
"""

from .offline_env import (
    MinecraftEnv,
    DiscreteAction,
    ObservationSpace,
    ActionSpace,
)
from .lemoncloud_env import (
    LemonCloudEnv,
    LiveEnvConfig,
)

__all__ = [
    'MinecraftEnv',
    'DiscreteAction',
    'ObservationSpace',
    'ActionSpace',
    'LemonCloudEnv',
    'LiveEnvConfig',
]
