"""
Environment module for Minecraft RL Bot.

This module provides the Minecraft environment simulation including:
- Block types and properties
- Action definitions
- Core environment logic

To plug in real Minecraft:
1. Keep the same observation/action interfaces
2. Replace MinecraftEnv with a wrapper around Malmo/MineRL
"""

from .blocks import BlockType, BLOCK_PROPERTIES, is_solid, is_transparent, get_block_hardness
from .actions import (
    DiscreteAction, ContinuousAction,
    MovementAction, CameraAction, InteractionAction, InventoryAction,
    create_action_space_info, sample_random_action,
    DISCRETE_ACTION_DIMS, CONTINUOUS_ACTION_DIM
)
from .core_env import MinecraftEnv, AgentState, Dimension

__all__ = [
    'BlockType',
    'BLOCK_PROPERTIES',
    'is_solid',
    'is_transparent',
    'get_block_hardness',
    'DiscreteAction',
    'ContinuousAction',
    'MovementAction',
    'CameraAction',
    'InteractionAction',
    'InventoryAction',
    'create_action_space_info',
    'sample_random_action',
    'DISCRETE_ACTION_DIMS',
    'CONTINUOUS_ACTION_DIM',
    'MinecraftEnv',
    'AgentState',
    'Dimension'
]
