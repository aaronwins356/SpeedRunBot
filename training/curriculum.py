"""
curriculum.py - Curriculum learning stages for Minecraft RL agent.

This module manages progressive difficulty scaling:
- Define curriculum stages with specific objectives
- Track progress and determine stage advancement
- Configure environment and rewards for each stage

Curriculum learning allows the agent to:
1. Master basic skills before complex tasks
2. Build on previous knowledge
3. Avoid the sparse reward problem
4. Achieve better final performance
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import IntEnum

from .reward import CurriculumStage, RewardConfig, create_stage_reward_config


@dataclass
class StageConfig:
    """
    Configuration for a curriculum stage.
    
    Attributes:
        stage: The curriculum stage enum value
        name: Human-readable name
        description: What the agent should learn
        objectives: List of objectives to complete
        success_threshold: Fraction of objectives needed to advance
        min_episodes: Minimum episodes before advancement
        max_episodes: Maximum episodes before forced advancement
        reward_config: Reward shaping for this stage
        env_config: Environment modifications for this stage
    """
    stage: CurriculumStage
    name: str
    description: str
    objectives: List[str] = field(default_factory=list)
    success_threshold: float = 0.8
    min_episodes: int = 100
    max_episodes: int = 1000
    reward_config: Optional[RewardConfig] = None
    env_config: Dict = field(default_factory=dict)


# Define all curriculum stages
CURRICULUM_STAGES = {
    CurriculumStage.BASIC_SURVIVAL: StageConfig(
        stage=CurriculumStage.BASIC_SURVIVAL,
        name="Basic Survival",
        description="Learn to survive: avoid damage, move around safely",
        objectives=[
            "survive_60_seconds",
            "avoid_fall_damage",
            "explore_100_blocks"
        ],
        success_threshold=0.7,
        min_episodes=50,
        max_episodes=500,
        env_config={
            'spawn_mobs': False,
            'day_only': True,
            'flat_terrain': True
        }
    ),
    
    CurriculumStage.RESOURCE_GATHERING: StageConfig(
        stage=CurriculumStage.RESOURCE_GATHERING,
        name="Resource Gathering",
        description="Learn to mine blocks and collect resources",
        objectives=[
            "mine_5_wood_logs",
            "mine_10_cobblestone",
            "collect_coal"
        ],
        success_threshold=0.75,
        min_episodes=100,
        max_episodes=800,
        env_config={
            'spawn_mobs': False,
            'day_only': True,
            'rich_resources': True
        }
    ),
    
    CurriculumStage.TOOL_CRAFTING: StageConfig(
        stage=CurriculumStage.TOOL_CRAFTING,
        name="Tool Crafting",
        description="Learn to craft tools and use crafting systems",
        objectives=[
            "craft_planks",
            "craft_wooden_pickaxe",
            "craft_stone_pickaxe"
        ],
        success_threshold=0.8,
        min_episodes=150,
        max_episodes=1000,
        env_config={
            'spawn_mobs': False,
            'nearby_crafting_table': True
        }
    ),
    
    CurriculumStage.NETHER_ACCESS: StageConfig(
        stage=CurriculumStage.NETHER_ACCESS,
        name="Nether Access",
        description="Learn to build and activate a Nether portal",
        objectives=[
            "mine_10_obsidian",
            "build_portal_frame",
            "enter_nether"
        ],
        success_threshold=0.7,
        min_episodes=200,
        max_episodes=2000,
        env_config={
            'nearby_lava': True,
            'nearby_water': True,
            'iron_available': True
        }
    ),
    
    CurriculumStage.BLAZE_HUNTING: StageConfig(
        stage=CurriculumStage.BLAZE_HUNTING,
        name="Blaze Hunting",
        description="Find Nether fortress and collect Blaze rods",
        objectives=[
            "find_fortress",
            "kill_3_blazes",
            "collect_blaze_rods"
        ],
        success_threshold=0.6,
        min_episodes=200,
        max_episodes=3000,
        env_config={
            'spawn_in_nether': True,
            'nearby_fortress': True
        }
    ),
    
    CurriculumStage.ENDER_PEARL_HUNT: StageConfig(
        stage=CurriculumStage.ENDER_PEARL_HUNT,
        name="Ender Pearl Hunt",
        description="Collect Ender Pearls through trading or hunting",
        objectives=[
            "find_endermen",
            "collect_12_ender_pearls",
            "survive_night"
        ],
        success_threshold=0.6,
        min_episodes=200,
        max_episodes=3000,
        env_config={
            'night_time': True,
            'spawn_endermen': True
        }
    ),
    
    CurriculumStage.END_PREPARATION: StageConfig(
        stage=CurriculumStage.END_PREPARATION,
        name="End Preparation",
        description="Craft Eyes of Ender and find the stronghold",
        objectives=[
            "craft_eyes_of_ender",
            "use_eye_to_find_stronghold",
            "locate_end_portal"
        ],
        success_threshold=0.7,
        min_episodes=150,
        max_episodes=2000,
        env_config={
            'has_blaze_powder': True,
            'has_ender_pearls': True
        }
    ),
    
    CurriculumStage.DRAGON_FIGHT: StageConfig(
        stage=CurriculumStage.DRAGON_FIGHT,
        name="Dragon Fight",
        description="Enter the End and defeat the Ender Dragon",
        objectives=[
            "enter_end",
            "destroy_crystals",
            "defeat_dragon"
        ],
        success_threshold=0.5,
        min_episodes=200,
        max_episodes=5000,
        env_config={
            'spawn_in_end': True,
            'has_equipment': True
        }
    ),
    
    CurriculumStage.FULL_GAME: StageConfig(
        stage=CurriculumStage.FULL_GAME,
        name="Full Game",
        description="Complete a full Minecraft speedrun",
        objectives=[
            "complete_game",
            "complete_in_time"
        ],
        success_threshold=0.1,  # Very hard
        min_episodes=500,
        max_episodes=100000,
        env_config={
            'full_game': True
        }
    )
}


class CurriculumManager:
    """
    Manager for curriculum learning progression.
    
    This class tracks agent progress across curriculum stages
    and determines when to advance to harder stages.
    
    Usage:
        curriculum = CurriculumManager()
        stage_config = curriculum.get_current_stage()
        curriculum.record_episode(success=True, metrics={...})
        if curriculum.should_advance():
            curriculum.advance_stage()
    """
    
    def __init__(
        self,
        start_stage: CurriculumStage = CurriculumStage.BASIC_SURVIVAL,
        auto_advance: bool = True
    ):
        """
        Initialize the curriculum manager.
        
        Args:
            start_stage: Initial curriculum stage
            auto_advance: Whether to automatically advance stages
        """
        self.current_stage = start_stage
        self.auto_advance = auto_advance
        
        # Progress tracking per stage
        self.stage_history: Dict[CurriculumStage, Dict] = {}
        for stage in CurriculumStage:
            self.stage_history[stage] = {
                'episodes': 0,
                'successes': 0,
                'objectives_completed': set(),
                'best_reward': float('-inf'),
                'avg_reward': 0.0,
                'reward_history': []
            }
        
        # Episode tracking for current stage
        self.current_episode_count = 0
    
    def get_current_stage(self) -> StageConfig:
        """Get the current stage configuration."""
        return CURRICULUM_STAGES[self.current_stage]
    
    def get_reward_config(self) -> RewardConfig:
        """Get the reward configuration for current stage."""
        config = CURRICULUM_STAGES[self.current_stage].reward_config
        if config is None:
            config = create_stage_reward_config(self.current_stage)
        return config
    
    def get_env_config(self) -> Dict:
        """Get environment configuration for current stage."""
        return CURRICULUM_STAGES[self.current_stage].env_config
    
    def record_episode(
        self,
        success: bool,
        total_reward: float,
        objectives_completed: Optional[List[str]] = None,
        metrics: Optional[Dict] = None
    ) -> None:
        """
        Record the results of an episode.
        
        Args:
            success: Whether the episode was successful
            total_reward: Total reward obtained
            objectives_completed: List of completed objectives
            metrics: Additional metrics
        """
        history = self.stage_history[self.current_stage]
        
        history['episodes'] += 1
        self.current_episode_count += 1
        
        if success:
            history['successes'] += 1
        
        if objectives_completed:
            history['objectives_completed'].update(objectives_completed)
        
        # Update reward statistics
        history['reward_history'].append(total_reward)
        if len(history['reward_history']) > 100:
            history['reward_history'] = history['reward_history'][-100:]
        
        history['best_reward'] = max(history['best_reward'], total_reward)
        history['avg_reward'] = np.mean(history['reward_history'])
        
        # Auto-advance if enabled
        if self.auto_advance and self.should_advance():
            self.advance_stage()
    
    def should_advance(self) -> bool:
        """
        Determine if the agent should advance to the next stage.
        
        Advancement criteria:
        1. Minimum episodes completed
        2. Success rate above threshold OR max episodes reached
        """
        if self.current_stage == CurriculumStage.FULL_GAME:
            return False  # Already at final stage
        
        stage_config = CURRICULUM_STAGES[self.current_stage]
        history = self.stage_history[self.current_stage]
        
        # Check minimum episodes
        if history['episodes'] < stage_config.min_episodes:
            return False
        
        # Check success rate
        if history['episodes'] > 0:
            success_rate = history['successes'] / history['episodes']
            if success_rate >= stage_config.success_threshold:
                return True
        
        # Check maximum episodes (forced advancement)
        if history['episodes'] >= stage_config.max_episodes:
            return True
        
        return False
    
    def advance_stage(self) -> bool:
        """
        Advance to the next curriculum stage.
        
        Returns:
            True if advanced, False if already at final stage
        """
        if self.current_stage == CurriculumStage.FULL_GAME:
            return False
        
        # Move to next stage
        next_stage = CurriculumStage(self.current_stage.value + 1)
        self.current_stage = next_stage
        self.current_episode_count = 0
        
        print(f"Advancing to curriculum stage: {CURRICULUM_STAGES[next_stage].name}")
        
        return True
    
    def set_stage(self, stage: CurriculumStage) -> None:
        """Manually set the curriculum stage."""
        self.current_stage = stage
        self.current_episode_count = 0
    
    def get_progress_report(self) -> Dict:
        """
        Get a summary of curriculum progress.
        
        Returns:
            Dictionary with progress information
        """
        return {
            'current_stage': self.current_stage.name,
            'stage_value': self.current_stage.value,
            'episode_count': self.current_episode_count,
            'stage_history': {
                stage.name: {
                    'episodes': history['episodes'],
                    'successes': history['successes'],
                    'success_rate': (
                        history['successes'] / max(1, history['episodes'])
                    ),
                    'best_reward': history['best_reward'],
                    'avg_reward': history['avg_reward'],
                    'objectives': list(history['objectives_completed'])
                }
                for stage, history in self.stage_history.items()
            }
        }
    
    def save_state(self) -> Dict:
        """Get serializable state for saving."""
        return {
            'current_stage': self.current_stage.value,
            'auto_advance': self.auto_advance,
            'current_episode_count': self.current_episode_count,
            'stage_history': {
                stage.value: {
                    'episodes': h['episodes'],
                    'successes': h['successes'],
                    'objectives_completed': list(h['objectives_completed']),
                    'best_reward': float(h['best_reward']) if h['best_reward'] != float('-inf') else None,
                    'avg_reward': float(h['avg_reward']),
                    'reward_history': [float(r) for r in h['reward_history']]
                }
                for stage, h in self.stage_history.items()
            }
        }
    
    def load_state(self, state: Dict) -> None:
        """Load state from saved data."""
        self.current_stage = CurriculumStage(state['current_stage'])
        self.auto_advance = state.get('auto_advance', True)
        self.current_episode_count = state.get('current_episode_count', 0)
        
        for stage_value, history_data in state.get('stage_history', {}).items():
            stage = CurriculumStage(int(stage_value))
            self.stage_history[stage] = {
                'episodes': history_data.get('episodes', 0),
                'successes': history_data.get('successes', 0),
                'objectives_completed': set(history_data.get('objectives_completed', [])),
                'best_reward': (
                    history_data['best_reward'] 
                    if history_data.get('best_reward') is not None 
                    else float('-inf')
                ),
                'avg_reward': history_data.get('avg_reward', 0.0),
                'reward_history': history_data.get('reward_history', [])
            }


def get_stage_description(stage: CurriculumStage) -> str:
    """Get a human-readable description of a curriculum stage."""
    config = CURRICULUM_STAGES[stage]
    return f"{config.name}: {config.description}"


def list_all_stages() -> List[Dict]:
    """List all curriculum stages with their details."""
    return [
        {
            'stage': stage.value,
            'name': config.name,
            'description': config.description,
            'objectives': config.objectives,
            'min_episodes': config.min_episodes,
            'max_episodes': config.max_episodes
        }
        for stage, config in CURRICULUM_STAGES.items()
    ]
