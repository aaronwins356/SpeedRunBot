"""
train.py - Main training loop for Minecraft RL agent.

This module provides the complete training pipeline:
- Environment interaction
- Policy optimization
- Curriculum management
- Logging and checkpointing

The training loop supports:
- Policy gradient methods (REINFORCE, A2C)
- Value function estimation
- Curriculum learning progression
- Periodic model saving

Usage:
    python -m minecraft_rl_bot.training.train --config config.yaml
"""

import numpy as np
import time
import json
import os
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from ..env import MinecraftEnv, DiscreteAction
from ..agent import Policy, PolicyConfig, ModelConfig, RolloutBuffer
from .reward import RewardShaper, RewardConfig, CurriculumStage
from .curriculum import CurriculumManager, CURRICULUM_STAGES
from ..utils.logger import TrainingLogger
from ..utils.helpers import load_config, set_seed


@dataclass
class TrainingConfig:
    """
    Configuration for the training loop.
    
    Attributes:
        num_episodes: Total episodes to train
        max_steps_per_episode: Maximum steps before truncation
        batch_size: Number of steps per optimization batch
        gamma: Discount factor for returns
        gae_lambda: Lambda for Generalized Advantage Estimation
        learning_rate: Optimizer learning rate
        value_loss_coef: Weight for value loss
        entropy_coef: Weight for entropy bonus
        max_grad_norm: Maximum gradient norm for clipping
        
        log_interval: Episodes between logging
        save_interval: Episodes between checkpoints
        eval_interval: Episodes between evaluation runs
        
        checkpoint_dir: Directory for saving checkpoints
        use_curriculum: Whether to use curriculum learning
        seed: Random seed
    """
    # Training parameters
    num_episodes: int = 10000
    max_steps_per_episode: int = 10000
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 0.0003
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Logging and saving
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 50
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    
    # Features
    use_curriculum: bool = True
    continuous_actions: bool = False
    seed: int = 42


class Trainer:
    """
    Main training class for the Minecraft RL agent.
    
    This class orchestrates:
    1. Environment interaction (collecting experience)
    2. Policy updates (learning from experience)
    3. Curriculum progression (increasing difficulty)
    4. Logging and checkpointing
    
    The training algorithm is based on Actor-Critic with
    Generalized Advantage Estimation (similar to A2C/PPO).
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        env: Optional[MinecraftEnv] = None,
        policy: Optional[Policy] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
            env: Environment instance (created if None)
            policy: Policy instance (created if None)
        """
        self.config = config if config is not None else TrainingConfig()
        
        # Set random seed
        set_seed(self.config.seed)
        
        # Create environment
        self.env = env if env is not None else MinecraftEnv(
            seed=self.config.seed,
            continuous_actions=self.config.continuous_actions
        )
        
        # Create policy
        if policy is None:
            model_config = ModelConfig(
                use_continuous=self.config.continuous_actions
            )
            policy_config = PolicyConfig(model_config=model_config)
            self.policy = Policy(policy_config)
        else:
            self.policy = policy
        
        # Create curriculum manager
        self.curriculum = CurriculumManager(
            auto_advance=self.config.use_curriculum
        )
        
        # Create reward shaper
        self.reward_shaper = RewardShaper(
            self.curriculum.get_reward_config()
        )
        
        # Create rollout buffer
        self.rollout_buffer = RolloutBuffer()
        
        # Create logger
        self.logger = TrainingLogger(
            log_dir=os.path.join(self.config.checkpoint_dir, "logs")
        )
        
        # Training state
        self.total_steps = 0
        self.total_episodes = 0
        self.best_reward = float('-inf')
        
        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def train(self) -> Dict:
        """
        Run the full training loop.
        
        Returns:
            Training summary statistics
        """
        print("Starting training...")
        print(f"  Episodes: {self.config.num_episodes}")
        print(f"  Curriculum: {self.config.use_curriculum}")
        print(f"  Checkpoint dir: {self.config.checkpoint_dir}")
        print()
        
        start_time = time.time()
        
        try:
            for episode in range(self.config.num_episodes):
                # Collect episode
                episode_stats = self._run_episode()
                self.total_episodes += 1
                
                # Update policy if buffer is full
                if len(self.rollout_buffer) >= self.config.batch_size:
                    update_stats = self._update_policy()
                    self.rollout_buffer.clear()
                else:
                    update_stats = {}
                
                # Record episode in curriculum
                self.curriculum.record_episode(
                    success=episode_stats.get('game_complete', False),
                    total_reward=episode_stats['total_reward'],
                    objectives_completed=episode_stats.get('objectives', [])
                )
                
                # Update reward shaper with new curriculum stage
                self.reward_shaper.set_curriculum_stage(
                    self.curriculum.current_stage
                )
                
                # Decay exploration
                self.policy.decay_epsilon()
                
                # Logging
                if episode % self.config.log_interval == 0:
                    self._log_progress(episode, episode_stats, update_stats)
                
                # Save checkpoint
                if episode % self.config.save_interval == 0 and episode > 0:
                    self._save_checkpoint(episode)
                
                # Track best reward
                if episode_stats['total_reward'] > self.best_reward:
                    self.best_reward = episode_stats['total_reward']
                    self._save_checkpoint(episode, is_best=True)
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        # Final save
        self._save_checkpoint(self.total_episodes, is_final=True)
        
        elapsed_time = time.time() - start_time
        
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'elapsed_time': elapsed_time,
            'curriculum_progress': self.curriculum.get_progress_report()
        }
    
    def _run_episode(self) -> Dict:
        """
        Run a single episode and collect experience.
        
        Returns:
            Episode statistics
        """
        # Reset environment and reward shaper
        observation, info = self.env.reset()
        self.reward_shaper.reset()
        
        episode_reward = 0.0
        episode_steps = 0
        episode_events = []
        
        for step in range(self.config.max_steps_per_episode):
            # Select action
            action, log_prob, value = self.policy.act(
                self._prepare_observation(observation)
            )
            
            # Execute action
            next_observation, env_reward, terminated, truncated, next_info = self.env.step(
                self._convert_action(action)
            )
            
            # Compute shaped reward
            shaped_reward = self.reward_shaper.compute_reward(
                info, next_info
            )
            total_reward = env_reward + shaped_reward
            
            # Store transition (raw observation without batch dimension)
            self.rollout_buffer.add(
                observation=observation,  # Store raw observation
                action=action,
                reward=total_reward,
                log_prob=log_prob[0] if isinstance(log_prob, np.ndarray) else log_prob,
                value=value[0] if isinstance(value, np.ndarray) else value,
                done=terminated or truncated
            )
            
            # Update counters
            episode_reward += total_reward
            episode_steps += 1
            self.total_steps += 1
            
            # Check for episode end
            if terminated or truncated:
                break
            
            observation = next_observation
            info = next_info
        
        return {
            'total_reward': episode_reward,
            'steps': episode_steps,
            'game_complete': info.get('game_complete', False),
            'health': info.get('health', 0),
            'day_count': info.get('day_count', 0),
            'stats': info.get('stats', {}),
            'objectives': episode_events
        }
    
    def _update_policy(self) -> Dict:
        """
        Update the policy using collected experience.
        
        Uses a simple policy gradient update with value baseline.
        
        Returns:
            Update statistics
        """
        # Get final value estimate for bootstrapping
        batch = self.rollout_buffer.get_batch()
        
        # Compute returns and advantages
        returns, advantages = self.rollout_buffer.compute_returns_and_advantages(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            last_value=0.0  # Assume episode ended
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Evaluate actions
        log_probs, values, entropy = self.policy.evaluate_actions(
            batch['observations'],
            batch['actions']
        )
        
        # Policy loss (negative because we want to maximize)
        policy_loss = -np.mean(log_probs * advantages)
        
        # Value loss
        value_loss = np.mean((values - returns) ** 2)
        
        # Entropy bonus (negative because we want to maximize entropy)
        entropy_loss = -np.mean(entropy)
        
        # Total loss
        total_loss = (
            policy_loss +
            self.config.value_loss_coef * value_loss +
            self.config.entropy_coef * entropy_loss
        )
        
        # Compute gradients and update
        # NOTE: This is a simplified update. In practice, you would use
        # an optimizer like Adam with proper gradient computation.
        self._simple_gradient_update(total_loss, batch)
        
        return {
            'policy_loss': float(policy_loss),
            'value_loss': float(value_loss),
            'entropy': float(-entropy_loss),
            'total_loss': float(total_loss)
        }
    
    def _simple_gradient_update(self, loss: float, batch: Dict) -> None:
        """
        Perform a simple gradient update.
        
        This is a placeholder for proper gradient computation.
        In production, you would use PyTorch autograd or JAX grad.
        
        For now, we use numerical gradient estimation for simplicity.
        """
        params = self.policy.get_parameters()
        lr = self.config.learning_rate
        
        # Simple parameter noise for exploration/learning
        # This is NOT proper gradient descent - just for demonstration
        for name, param in params.items():
            noise = np.random.randn(*param.shape) * lr * 0.1
            params[name] = param - noise
        
        self.policy.set_parameters(params)
    
    def _prepare_observation(self, obs: Dict) -> Dict:
        """Prepare observation for policy input."""
        return {
            'blocks': obs['blocks'][np.newaxis, ...],
            'inventory': obs['inventory'][np.newaxis, ...],
            'agent_state': obs['agent_state'][np.newaxis, ...]
        }
    
    def _convert_action(self, action: Dict) -> DiscreteAction:
        """Convert policy output to environment action."""
        if isinstance(action, dict):
            return DiscreteAction(
                movement=int(action.get('movement', [0])[0]),
                camera=int(action.get('camera', [0])[0]),
                interaction=int(action.get('interaction', [0])[0]),
                inventory=int(action.get('inventory', [0])[0])
            )
        return action
    
    def _log_progress(
        self,
        episode: int,
        episode_stats: Dict,
        update_stats: Dict
    ) -> None:
        """Log training progress."""
        self.logger.log_episode(
            episode=episode,
            reward=episode_stats['total_reward'],
            steps=episode_stats['steps'],
            info={
                'curriculum_stage': self.curriculum.current_stage.name,
                'epsilon': self.policy.epsilon,
                **update_stats
            }
        )
        
        # Console output
        print(f"Episode {episode:5d} | "
              f"Reward: {episode_stats['total_reward']:8.2f} | "
              f"Steps: {episode_stats['steps']:5d} | "
              f"Stage: {self.curriculum.current_stage.name}")
    
    def _save_checkpoint(
        self,
        episode: int,
        is_best: bool = False,
        is_final: bool = False
    ) -> None:
        """Save a training checkpoint."""
        if is_best:
            filename = "best_model.npz"
        elif is_final:
            filename = "final_model.npz"
        else:
            filename = f"checkpoint_{episode}.npz"
        
        path = os.path.join(self.config.checkpoint_dir, filename)
        
        # Save model
        self.policy.save(path)
        
        # Save training state
        state = {
            'episode': episode,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'curriculum': self.curriculum.save_state(),
            'config': {
                'num_episodes': self.config.num_episodes,
                'batch_size': self.config.batch_size,
                'gamma': self.config.gamma,
                'learning_rate': self.config.learning_rate
            }
        }
        
        state_path = path.replace('.npz', '_state.json')
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Saved checkpoint: {filename}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint."""
        # Load model
        self.policy.load(path)
        
        # Load training state
        state_path = path.replace('.npz', '_state.json')
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            self.total_episodes = state.get('episode', 0)
            self.total_steps = state.get('total_steps', 0)
            self.best_reward = state.get('best_reward', float('-inf'))
            
            if 'curriculum' in state:
                self.curriculum.load_state(state['curriculum'])
        
        print(f"Loaded checkpoint: {path}")
    
    def evaluate(self, num_episodes: int = 10) -> Dict:
        """
        Evaluate the current policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation statistics
        """
        self.policy.set_training(False)
        
        rewards = []
        steps = []
        completions = 0
        
        for _ in range(num_episodes):
            observation, info = self.env.reset()
            episode_reward = 0.0
            episode_steps = 0
            
            for _ in range(self.config.max_steps_per_episode):
                action, _, _ = self.policy.act(
                    self._prepare_observation(observation),
                    deterministic=True
                )
                
                observation, reward, terminated, truncated, info = self.env.step(
                    self._convert_action(action)
                )
                
                episode_reward += reward
                episode_steps += 1
                
                if terminated or truncated:
                    break
            
            rewards.append(episode_reward)
            steps.append(episode_steps)
            if info.get('game_complete', False):
                completions += 1
        
        self.policy.set_training(True)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_steps': np.mean(steps),
            'completion_rate': completions / num_episodes
        }


def main():
    """Main entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Minecraft RL Agent")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of episodes to train')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    args = parser.parse_args()
    
    # Load configuration
    config_dict = load_config(args.config)
    config = TrainingConfig(**config_dict) if config_dict else TrainingConfig()
    
    # Override from command line
    if args.episodes:
        config.num_episodes = args.episodes
    if args.seed:
        config.seed = args.seed
    
    # Create trainer
    trainer = Trainer(config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train
    results = trainer.train()
    
    # Print results
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Total Episodes: {results['total_episodes']}")
    print(f"Total Steps: {results['total_steps']}")
    print(f"Best Reward: {results['best_reward']:.2f}")
    print(f"Elapsed Time: {results['elapsed_time']:.1f} seconds")
    print("\nCurriculum Progress:")
    for stage, info in results['curriculum_progress']['stage_history'].items():
        if info['episodes'] > 0:
            print(f"  {stage}: {info['episodes']} episodes, "
                  f"{info['success_rate']:.1%} success rate")


if __name__ == "__main__":
    main()
