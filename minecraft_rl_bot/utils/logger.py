"""
logger.py - Logging utilities for Minecraft RL agent training.

This module provides logging functionality for:
- Training progress (rewards, losses, etc.)
- Episode statistics
- Curriculum progression
- Action distributions

Logs are stored in both text format (for human reading) and
JSON format (for analysis and visualization).
"""

import json
import os
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime


@dataclass
class LogEntry:
    """A single log entry."""
    timestamp: float
    episode: int
    data: Dict[str, Any]


class TrainingLogger:
    """
    Logger for tracking training progress.
    
    Features:
    - Episode-level logging (rewards, steps, etc.)
    - Periodic summary statistics
    - JSON export for analysis
    - Console output formatting
    
    Usage:
        logger = TrainingLogger(log_dir="logs")
        logger.log_episode(episode=1, reward=10.5, steps=100)
        logger.save()
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to store log files
            experiment_name: Name for this training run
            verbose: Whether to print to console
        """
        self.log_dir = log_dir
        self.verbose = verbose
        
        # Generate experiment name if not provided
        if experiment_name is None:
            experiment_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        
        # Create log directory
        self.run_dir = os.path.join(log_dir, experiment_name)
        Path(self.run_dir).mkdir(parents=True, exist_ok=True)
        
        # Log storage
        self.episode_logs: List[LogEntry] = []
        self.step_logs: List[LogEntry] = []
        
        # Running statistics
        self.stats = {
            'total_episodes': 0,
            'total_steps': 0,
            'total_reward': 0.0,
            'best_reward': float('-inf'),
            'worst_reward': float('inf'),
            'recent_rewards': [],  # Last 100 rewards
            'start_time': time.time()
        }
        
        # Open text log file
        self.text_log_path = os.path.join(self.run_dir, "training.log")
        self._write_header()
    
    def _write_header(self) -> None:
        """Write header to text log file."""
        with open(self.text_log_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"Minecraft RL Agent Training Log\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n\n")
    
    def log_episode(
        self,
        episode: int,
        reward: float,
        steps: int,
        info: Optional[Dict] = None
    ) -> None:
        """
        Log an episode completion.
        
        Args:
            episode: Episode number
            reward: Total episode reward
            steps: Number of steps in episode
            info: Additional information to log
        """
        timestamp = time.time()
        
        # Create log entry
        data = {
            'reward': reward,
            'steps': steps,
            'elapsed_time': timestamp - self.stats['start_time']
        }
        if info:
            data.update(info)
        
        entry = LogEntry(timestamp=timestamp, episode=episode, data=data)
        self.episode_logs.append(entry)
        
        # Update statistics
        self.stats['total_episodes'] += 1
        self.stats['total_steps'] += steps
        self.stats['total_reward'] += reward
        self.stats['best_reward'] = max(self.stats['best_reward'], reward)
        self.stats['worst_reward'] = min(self.stats['worst_reward'], reward)
        
        self.stats['recent_rewards'].append(reward)
        if len(self.stats['recent_rewards']) > 100:
            self.stats['recent_rewards'] = self.stats['recent_rewards'][-100:]
        
        # Write to text log
        self._write_episode_log(episode, reward, steps, info)
    
    def _write_episode_log(
        self,
        episode: int,
        reward: float,
        steps: int,
        info: Optional[Dict]
    ) -> None:
        """Write episode to text log file."""
        with open(self.text_log_path, 'a') as f:
            f.write(f"Episode {episode:6d} | "
                    f"Reward: {reward:10.2f} | "
                    f"Steps: {steps:6d}")
            
            if info:
                info_str = " | ".join(f"{k}: {v}" for k, v in info.items()
                                      if k not in ['reward', 'steps'])
                if info_str:
                    f.write(f" | {info_str}")
            
            f.write("\n")
    
    def log_step(
        self,
        episode: int,
        step: int,
        action: Any,
        reward: float,
        info: Optional[Dict] = None
    ) -> None:
        """
        Log a single step (use sparingly - can be expensive).
        
        Args:
            episode: Current episode
            step: Current step within episode
            action: Action taken
            reward: Reward received
            info: Additional information
        """
        timestamp = time.time()
        
        data = {
            'step': step,
            'action': str(action),
            'reward': reward
        }
        if info:
            data.update(info)
        
        entry = LogEntry(timestamp=timestamp, episode=episode, data=data)
        self.step_logs.append(entry)
        
        # Limit step log size
        if len(self.step_logs) > 10000:
            self.step_logs = self.step_logs[-5000:]
    
    def log_update(
        self,
        episode: int,
        losses: Dict[str, float],
        info: Optional[Dict] = None
    ) -> None:
        """
        Log a policy update.
        
        Args:
            episode: Current episode
            losses: Dictionary of loss values
            info: Additional information
        """
        with open(self.text_log_path, 'a') as f:
            loss_str = " | ".join(f"{k}: {v:.6f}" for k, v in losses.items())
            f.write(f"  Update @ Episode {episode}: {loss_str}\n")
    
    def log_curriculum_change(
        self,
        old_stage: str,
        new_stage: str,
        episode: int
    ) -> None:
        """Log a curriculum stage change."""
        with open(self.text_log_path, 'a') as f:
            f.write(f"\n{'='*40}\n")
            f.write(f"CURRICULUM ADVANCEMENT @ Episode {episode}\n")
            f.write(f"  {old_stage} -> {new_stage}\n")
            f.write(f"{'='*40}\n\n")
        
        if self.verbose:
            print(f"\n🎯 Curriculum Advanced: {old_stage} -> {new_stage}\n")
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        recent_rewards = self.stats['recent_rewards']
        
        return {
            'total_episodes': self.stats['total_episodes'],
            'total_steps': self.stats['total_steps'],
            'total_reward': self.stats['total_reward'],
            'best_reward': self.stats['best_reward'],
            'worst_reward': self.stats['worst_reward'],
            'mean_reward': (
                sum(recent_rewards) / len(recent_rewards)
                if recent_rewards else 0
            ),
            'elapsed_time': time.time() - self.stats['start_time']
        }
    
    def save(self) -> None:
        """Save all logs to files."""
        # Save episode logs as JSON
        episode_log_path = os.path.join(self.run_dir, "episodes.json")
        with open(episode_log_path, 'w') as f:
            json.dump([
                {
                    'timestamp': e.timestamp,
                    'episode': e.episode,
                    **e.data
                }
                for e in self.episode_logs
            ], f, indent=2)
        
        # Save summary
        summary_path = os.path.join(self.run_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
        
        if self.verbose:
            print(f"Logs saved to: {self.run_dir}")
    
    def close(self) -> None:
        """Close the logger and save all data."""
        self.save()
        
        # Write footer to text log
        with open(self.text_log_path, 'a') as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Training Complete: {datetime.now().isoformat()}\n")
            f.write(f"Total Episodes: {self.stats['total_episodes']}\n")
            f.write(f"Total Steps: {self.stats['total_steps']}\n")
            f.write(f"Best Reward: {self.stats['best_reward']:.2f}\n")
            f.write("=" * 60 + "\n")


class ActionLogger:
    """
    Logger for tracking action distributions and choices.
    
    Useful for debugging agent behavior and understanding
    what actions the agent prefers.
    """
    
    def __init__(self):
        """Initialize action logger."""
        self.action_counts: Dict[str, Dict[int, int]] = {
            'movement': {},
            'camera': {},
            'interaction': {},
            'inventory': {}
        }
        self.total_actions = 0
    
    def log_action(self, action: Dict[str, int]) -> None:
        """
        Log an action selection.
        
        Args:
            action: Dictionary mapping action type to action index
        """
        for action_type, action_idx in action.items():
            if action_type in self.action_counts:
                if action_idx not in self.action_counts[action_type]:
                    self.action_counts[action_type][action_idx] = 0
                self.action_counts[action_type][action_idx] += 1
        
        self.total_actions += 1
    
    def get_distribution(self, action_type: str) -> Dict[int, float]:
        """
        Get the action distribution for a type.
        
        Args:
            action_type: Type of action ('movement', 'camera', etc.)
            
        Returns:
            Dictionary mapping action index to frequency
        """
        counts = self.action_counts.get(action_type, {})
        total = sum(counts.values())
        
        if total == 0:
            return {}
        
        return {k: v / total for k, v in counts.items()}
    
    def get_summary(self) -> Dict:
        """Get summary of action distributions."""
        return {
            action_type: self.get_distribution(action_type)
            for action_type in self.action_counts
        }
    
    def reset(self) -> None:
        """Reset action counts."""
        for action_type in self.action_counts:
            self.action_counts[action_type] = {}
        self.total_actions = 0


class RewardVisualizer:
    """
    Simple text-based reward visualization.
    
    Provides ASCII charts for viewing reward progress
    without external dependencies.
    """
    
    def __init__(self, width: int = 60, height: int = 20):
        """
        Initialize visualizer.
        
        Args:
            width: Chart width in characters
            height: Chart height in lines
        """
        self.width = width
        self.height = height
        self.rewards: List[float] = []
    
    def add_reward(self, reward: float) -> None:
        """Add a reward value."""
        self.rewards.append(reward)
    
    def render(self) -> str:
        """
        Render an ASCII chart of rewards.
        
        Returns:
            String containing ASCII chart
        """
        if not self.rewards:
            return "No data to display"
        
        # Sample rewards to fit width
        if len(self.rewards) > self.width:
            step = len(self.rewards) / self.width
            sampled = [
                self.rewards[int(i * step)]
                for i in range(self.width)
            ]
        else:
            sampled = self.rewards
        
        # Find range
        min_r = min(sampled)
        max_r = max(sampled)
        range_r = max_r - min_r if max_r != min_r else 1
        
        # Create chart
        lines = []
        lines.append(f"Reward over episodes (n={len(self.rewards)})")
        lines.append(f"Max: {max_r:.2f}  Min: {min_r:.2f}  "
                    f"Avg: {sum(self.rewards)/len(self.rewards):.2f}")
        lines.append("-" * (self.width + 2))
        
        for row in range(self.height - 1, -1, -1):
            threshold = min_r + (row / self.height) * range_r
            line = "|"
            for val in sampled:
                if val >= threshold:
                    line += "█"
                else:
                    line += " "
            line += "|"
            lines.append(line)
        
        lines.append("-" * (self.width + 2))
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear reward history."""
        self.rewards.clear()
