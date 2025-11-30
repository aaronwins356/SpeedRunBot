"""
Training module for Minecraft RL Bot.

This module provides the training loop and related components:
- Trainer: Main training orchestration with PyTorch
- RewardShaper: Reward signal construction
- CurriculumManager: Progressive difficulty scaling
"""

from .reward import (
    RewardConfig,
    RewardShaper,
    CurriculumStage,
    REWARD_VALUES,
    create_stage_reward_config
)
from .curriculum import (
    CurriculumManager,
    StageConfig,
    CURRICULUM_STAGES,
    get_stage_description,
    list_all_stages
)
from .train import (
    TrainingConfig,
    Trainer,
    main
)

__all__ = [
    'RewardConfig',
    'RewardShaper',
    'CurriculumStage',
    'REWARD_VALUES',
    'create_stage_reward_config',
    'CurriculumManager',
    'StageConfig',
    'CURRICULUM_STAGES',
    'get_stage_description',
    'list_all_stages',
    'TrainingConfig',
    'Trainer',
    'main'
]
