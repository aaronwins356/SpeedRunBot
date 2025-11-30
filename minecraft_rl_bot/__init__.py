"""
Minecraft RL Bot - A reinforcement learning agent for Minecraft.

This package provides a complete RL training framework for creating
agents that can play Minecraft, with the goal of beating the game
(reaching the end credits).

Key Features:
- Mock Minecraft environment for training
- Lightweight neural network models (CPU-friendly)
- Curriculum learning for progressive skill acquisition
- Reward shaping for effective learning

Modules:
- env: Environment simulation (blocks, actions, world)
- agent: Neural network models and policies
- training: Training loop, rewards, curriculum
- utils: Logging and helper utilities

Quick Start:
    from minecraft_rl_bot import MinecraftEnv, Trainer, TrainingConfig
    
    # Create environment
    env = MinecraftEnv(seed=42)
    
    # Create trainer
    config = TrainingConfig(num_episodes=1000)
    trainer = Trainer(config, env=env)
    
    # Train
    results = trainer.train()
"""

__version__ = "0.1.0"
__author__ = "Minecraft RL Bot Team"

# Import main components for easy access
from .env import (
    MinecraftEnv,
    BlockType,
    DiscreteAction,
    ContinuousAction
)
from .agent import (
    Policy,
    PolicyConfig,
    ModelConfig,
    MinecraftPolicy
)
from .training import (
    Trainer,
    TrainingConfig,
    RewardShaper,
    CurriculumManager,
    CurriculumStage
)
from .utils import (
    TrainingLogger,
    set_seed,
    load_config
)

__all__ = [
    # Version info
    '__version__',
    '__author__',
    # Environment
    'MinecraftEnv',
    'BlockType',
    'DiscreteAction',
    'ContinuousAction',
    # Agent
    'Policy',
    'PolicyConfig',
    'ModelConfig',
    'MinecraftPolicy',
    # Training
    'Trainer',
    'TrainingConfig',
    'RewardShaper',
    'CurriculumManager',
    'CurriculumStage',
    # Utils
    'TrainingLogger',
    'set_seed',
    'load_config'
]
