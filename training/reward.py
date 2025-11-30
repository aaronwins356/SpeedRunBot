"""
reward.py - Reward shaping logic for Minecraft RL agent.

This module provides reward signals that guide the agent's learning:
- Positive rewards for progress (mining, crafting, exploration)
- Negative rewards only for damage taken
- Curriculum-specific reward tuning

The reward system is designed to:
1. Encourage survival
2. Promote resource gathering
3. Guide progression through game stages
4. Provide dense signals for learning

REWARD PHILOSOPHY:
- Only penalize damage (not idle time or inefficiency)
- Reward milestones more than incremental progress
- Scale rewards based on curriculum stage
"""

import numpy as np
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import IntEnum


class CurriculumStage(IntEnum):
    """
    Curriculum stages for progressive learning.
    
    Each stage focuses on different skills and has
    different reward weightings.
    """
    BASIC_SURVIVAL = 0      # Don't die, move around
    RESOURCE_GATHERING = 1  # Mine blocks, collect items
    TOOL_CRAFTING = 2       # Craft tools, use crafting table
    NETHER_ACCESS = 3       # Build portal, enter Nether
    BLAZE_HUNTING = 4       # Find fortress, kill Blazes
    ENDER_PEARL_HUNT = 5    # Trade/find Ender Pearls
    END_PREPARATION = 6     # Craft Eyes, find stronghold
    DRAGON_FIGHT = 7        # Enter End, defeat dragon
    FULL_GAME = 8           # Complete speedrun


@dataclass
class RewardConfig:
    """
    Configuration for reward shaping.
    
    Attributes:
        curriculum_stage: Current curriculum stage
        damage_penalty_weight: Multiplier for damage penalties
        exploration_weight: Multiplier for exploration rewards
        milestone_bonus: Multiplier for milestone rewards
        time_penalty: Small penalty per tick to encourage efficiency
        enable_shaping: Whether to use shaped rewards
    """
    curriculum_stage: CurriculumStage = CurriculumStage.BASIC_SURVIVAL
    damage_penalty_weight: float = 2.0
    exploration_weight: float = 0.01
    milestone_bonus: float = 10.0
    time_penalty: float = 0.0  # Disabled by default
    enable_shaping: bool = True


# Maximum damage possible in a single tick (for reward normalization)
# Minecraft typically caps damage at around 3-4 hearts per tick from normal sources
MAX_DAMAGE_PER_TICK = 3.0

# Reward values for different events
REWARD_VALUES = {
    # Resource gathering (positive)
    'mine_wood_log': 1.0,
    'mine_cobblestone': 0.5,
    'mine_coal': 2.0,
    'mine_iron_ore': 5.0,
    'mine_gold_ore': 3.0,
    'mine_diamond': 20.0,
    'mine_obsidian': 5.0,
    
    # Crafting (positive)
    'craft_wood_planks': 0.5,
    'craft_sticks': 0.5,
    'craft_wooden_pickaxe': 3.0,
    'craft_stone_pickaxe': 5.0,
    'craft_iron_pickaxe': 10.0,
    'craft_diamond_pickaxe': 15.0,
    'craft_furnace': 5.0,
    'craft_crafting_table': 3.0,
    
    # Progression milestones (positive)
    'survive_day': 5.0,
    'enter_nether': 50.0,
    'find_fortress': 30.0,
    'kill_blaze': 20.0,
    'obtain_blaze_rod': 25.0,
    'obtain_ender_pearl': 15.0,
    'craft_eye_of_ender': 20.0,
    'find_stronghold': 40.0,
    'enter_end': 100.0,
    'kill_dragon': 500.0,
    'enter_portal': 1000.0,  # Game complete!
    
    # Damage (negative - ONLY penalty source)
    'damage_fall': -2.0,
    'damage_mob': -3.0,
    'damage_lava': -5.0,
    'damage_fire': -3.0,
    'damage_drown': -2.0,
    'death': -100.0,
}

# Stage-specific reward multipliers
STAGE_MULTIPLIERS = {
    CurriculumStage.BASIC_SURVIVAL: {
        'survival': 2.0,
        'exploration': 1.0,
        'gathering': 0.5,
        'crafting': 0.3,
        'progression': 0.1
    },
    CurriculumStage.RESOURCE_GATHERING: {
        'survival': 1.0,
        'exploration': 0.5,
        'gathering': 2.0,
        'crafting': 0.5,
        'progression': 0.2
    },
    CurriculumStage.TOOL_CRAFTING: {
        'survival': 1.0,
        'exploration': 0.3,
        'gathering': 1.0,
        'crafting': 2.0,
        'progression': 0.5
    },
    CurriculumStage.NETHER_ACCESS: {
        'survival': 1.0,
        'exploration': 0.5,
        'gathering': 0.5,
        'crafting': 1.0,
        'progression': 2.0
    },
    CurriculumStage.BLAZE_HUNTING: {
        'survival': 1.5,
        'exploration': 0.5,
        'gathering': 0.3,
        'crafting': 0.3,
        'progression': 2.0
    },
    CurriculumStage.ENDER_PEARL_HUNT: {
        'survival': 1.0,
        'exploration': 1.0,
        'gathering': 0.5,
        'crafting': 0.5,
        'progression': 2.0
    },
    CurriculumStage.END_PREPARATION: {
        'survival': 1.0,
        'exploration': 1.0,
        'gathering': 0.3,
        'crafting': 1.0,
        'progression': 2.0
    },
    CurriculumStage.DRAGON_FIGHT: {
        'survival': 2.0,
        'exploration': 0.3,
        'gathering': 0.1,
        'crafting': 0.1,
        'progression': 3.0
    },
    CurriculumStage.FULL_GAME: {
        'survival': 1.0,
        'exploration': 0.5,
        'gathering': 0.5,
        'crafting': 0.5,
        'progression': 1.5
    }
}


class RewardShaper:
    """
    Reward shaping system for Minecraft RL.
    
    This class tracks agent progress and computes shaped rewards
    based on the current curriculum stage and events.
    
    Usage:
        shaper = RewardShaper(config)
        reward = shaper.compute_reward(prev_info, curr_info, events)
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        """
        Initialize the reward shaper.
        
        Args:
            config: Reward configuration
        """
        self.config = config if config is not None else RewardConfig()
        
        # Track achieved milestones (no repeat rewards)
        self.achieved_milestones: Set[str] = set()
        
        # Track best progress for incremental rewards
        self.best_stats = {
            'max_health': 20.0,
            'blocks_mined': 0,
            'items_crafted': 0,
            'distance_traveled': 0.0,
            'days_survived': 0
        }
    
    def reset(self) -> None:
        """Reset tracking for a new episode."""
        self.achieved_milestones.clear()
        self.best_stats = {
            'max_health': 20.0,
            'blocks_mined': 0,
            'items_crafted': 0,
            'distance_traveled': 0.0,
            'days_survived': 0
        }
    
    def compute_reward(
        self,
        prev_info: Dict,
        curr_info: Dict,
        events: Optional[List[str]] = None
    ) -> float:
        """
        Compute the shaped reward for a transition.
        
        Args:
            prev_info: Environment info before action
            curr_info: Environment info after action
            events: List of events that occurred (optional)
            
        Returns:
            Shaped reward value
        """
        if not self.config.enable_shaping:
            return 0.0
        
        reward = 0.0
        stage = self.config.curriculum_stage
        multipliers = STAGE_MULTIPLIERS[stage]
        
        # 1. Damage penalty (ONLY negative reward source)
        damage_reward = self._compute_damage_reward(prev_info, curr_info)
        reward += damage_reward * multipliers['survival'] * self.config.damage_penalty_weight
        
        # 2. Exploration reward
        exploration_reward = self._compute_exploration_reward(prev_info, curr_info)
        reward += exploration_reward * multipliers['exploration'] * self.config.exploration_weight
        
        # 3. Gathering reward
        gathering_reward = self._compute_gathering_reward(prev_info, curr_info, events)
        reward += gathering_reward * multipliers['gathering']
        
        # 4. Crafting reward
        crafting_reward = self._compute_crafting_reward(prev_info, curr_info, events)
        reward += crafting_reward * multipliers['crafting']
        
        # 5. Progression milestones
        milestone_reward = self._compute_milestone_reward(prev_info, curr_info, events)
        reward += milestone_reward * multipliers['progression'] * self.config.milestone_bonus
        
        # 6. Time penalty (optional, disabled by default)
        if self.config.time_penalty > 0:
            reward -= self.config.time_penalty
        
        return reward
    
    def _compute_damage_reward(
        self,
        prev_info: Dict,
        curr_info: Dict
    ) -> float:
        """Compute penalty for taking damage."""
        prev_health = prev_info.get('health', 20)
        curr_health = curr_info.get('health', 20)
        
        damage = prev_health - curr_health
        
        if damage <= 0:
            return 0.0
        
        # Death is extra bad
        if curr_health <= 0:
            return REWARD_VALUES['death']
        
        # Scale by damage amount (normalized by max typical damage per tick)
        return REWARD_VALUES['damage_mob'] * damage / MAX_DAMAGE_PER_TICK
    
    def _compute_exploration_reward(
        self,
        prev_info: Dict,
        curr_info: Dict
    ) -> float:
        """Compute reward for exploring new areas."""
        prev_pos = np.array(prev_info.get('position', [0, 0, 0]))
        curr_pos = np.array(curr_info.get('position', [0, 0, 0]))
        
        distance = np.linalg.norm(curr_pos - prev_pos)
        
        # Track total distance for milestone purposes
        self.best_stats['distance_traveled'] += distance
        
        return distance
    
    def _compute_gathering_reward(
        self,
        prev_info: Dict,
        curr_info: Dict,
        events: Optional[List[str]] = None
    ) -> float:
        """Compute reward for gathering resources."""
        reward = 0.0
        
        # Check stats changes
        prev_stats = prev_info.get('stats', {})
        curr_stats = curr_info.get('stats', {})
        
        blocks_mined_diff = (
            curr_stats.get('blocks_mined', 0) - 
            prev_stats.get('blocks_mined', 0)
        )
        
        if blocks_mined_diff > 0:
            # Base reward for mining
            reward += blocks_mined_diff * 0.5
            self.best_stats['blocks_mined'] = curr_stats.get('blocks_mined', 0)
        
        # Process specific events
        if events:
            for event in events:
                if event.startswith('mine_'):
                    reward += REWARD_VALUES.get(event, 0.5)
        
        return reward
    
    def _compute_crafting_reward(
        self,
        prev_info: Dict,
        curr_info: Dict,
        events: Optional[List[str]] = None
    ) -> float:
        """Compute reward for crafting items."""
        reward = 0.0
        
        prev_stats = prev_info.get('stats', {})
        curr_stats = curr_info.get('stats', {})
        
        items_crafted_diff = (
            curr_stats.get('items_crafted', 0) - 
            prev_stats.get('items_crafted', 0)
        )
        
        if items_crafted_diff > 0:
            reward += items_crafted_diff * 2.0
            self.best_stats['items_crafted'] = curr_stats.get('items_crafted', 0)
        
        # Process specific events
        if events:
            for event in events:
                if event.startswith('craft_'):
                    reward += REWARD_VALUES.get(event, 2.0)
        
        return reward
    
    def _compute_milestone_reward(
        self,
        prev_info: Dict,
        curr_info: Dict,
        events: Optional[List[str]] = None
    ) -> float:
        """Compute reward for achieving milestones."""
        reward = 0.0
        
        # Check dimension changes
        if not prev_info.get('in_nether') and curr_info.get('in_nether'):
            if 'enter_nether' not in self.achieved_milestones:
                reward += REWARD_VALUES['enter_nether']
                self.achieved_milestones.add('enter_nether')
        
        if not prev_info.get('in_end') and curr_info.get('in_end'):
            if 'enter_end' not in self.achieved_milestones:
                reward += REWARD_VALUES['enter_end']
                self.achieved_milestones.add('enter_end')
        
        # Dragon defeat
        if not prev_info.get('dragon_defeated') and curr_info.get('dragon_defeated'):
            if 'kill_dragon' not in self.achieved_milestones:
                reward += REWARD_VALUES['kill_dragon']
                self.achieved_milestones.add('kill_dragon')
        
        # Game complete
        if curr_info.get('game_complete'):
            if 'enter_portal' not in self.achieved_milestones:
                reward += REWARD_VALUES['enter_portal']
                self.achieved_milestones.add('enter_portal')
        
        # Day survival
        prev_days = prev_info.get('day_count', 0)
        curr_days = curr_info.get('day_count', 0)
        if curr_days > prev_days:
            days_survived = curr_days - self.best_stats['days_survived']
            if days_survived > 0:
                reward += REWARD_VALUES['survive_day'] * days_survived
                self.best_stats['days_survived'] = curr_days
        
        # Process specific events
        if events:
            milestone_events = [
                'find_fortress', 'kill_blaze', 'obtain_blaze_rod',
                'obtain_ender_pearl', 'craft_eye_of_ender', 'find_stronghold'
            ]
            for event in events:
                if event in milestone_events and event not in self.achieved_milestones:
                    reward += REWARD_VALUES.get(event, 10.0)
                    self.achieved_milestones.add(event)
        
        return reward
    
    def set_curriculum_stage(self, stage: CurriculumStage) -> None:
        """Update the curriculum stage."""
        self.config.curriculum_stage = stage
    
    def get_stats(self) -> Dict:
        """Get current tracking statistics."""
        return {
            'achieved_milestones': list(self.achieved_milestones),
            'best_stats': self.best_stats.copy()
        }


def create_stage_reward_config(stage: CurriculumStage) -> RewardConfig:
    """
    Create a RewardConfig optimized for a specific curriculum stage.
    
    Args:
        stage: Target curriculum stage
        
    Returns:
        Configured RewardConfig instance
    """
    config = RewardConfig(curriculum_stage=stage)
    
    # Adjust weights based on stage
    if stage == CurriculumStage.BASIC_SURVIVAL:
        config.damage_penalty_weight = 3.0
        config.exploration_weight = 0.02
    elif stage == CurriculumStage.DRAGON_FIGHT:
        config.damage_penalty_weight = 2.0
        config.milestone_bonus = 20.0
    elif stage == CurriculumStage.FULL_GAME:
        config.time_penalty = 0.001  # Small penalty to encourage speed
    
    return config
