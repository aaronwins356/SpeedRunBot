"""
Integration module for Minecraft RL Bot.

This module provides integration with external Minecraft clients:
- MinecraftClient: Abstract interface for Minecraft connectivity
"""

from .mc_client import MinecraftClient, ClientConfig

__all__ = [
    'MinecraftClient',
    'ClientConfig',
]
