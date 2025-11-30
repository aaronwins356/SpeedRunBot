"""
policy.py - Policy wrapper and action selection for Minecraft RL agent.

This module provides high-level policy interfaces that:
- Wrap the neural network model
- Handle action selection and exploration
- Manage policy updates during training

The policy is the main interface between the training loop and the model.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from .model import MinecraftPolicy, ModelConfig, create_model


@dataclass
class PolicyConfig:
    """
    Configuration for the policy wrapper.
    
    Attributes:
        model_config: Configuration for the underlying model
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Decay rate for epsilon
        use_entropy_bonus: Whether to add entropy bonus to loss
        entropy_coef: Coefficient for entropy bonus
    """
    model_config: ModelConfig = None
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    use_entropy_bonus: bool = True
    entropy_coef: float = 0.01
    
    def __post_init__(self):
        if self.model_config is None:
            self.model_config = ModelConfig()


class Policy:
    """
    High-level policy class for the Minecraft RL agent.
    
    This class wraps the neural network model and provides:
    - Epsilon-greedy exploration
    - Action sampling with temperature
    - Batch action selection for training
    - Device management (CPU/GPU)
    
    Usage:
        policy = Policy(config)
        action, log_prob, value = policy.act(observation)
    """
    
    def __init__(self, config: Optional[PolicyConfig] = None):
        """
        Initialize the policy.
        
        Args:
            config: Policy configuration
        """
        self.config = config if config is not None else PolicyConfig()
        
        # Determine device (CPU for low-resource hardware)
        self.device = torch.device('cpu')
        
        # Create model
        self.model = create_model(self.config.model_config)
        self.model.to(self.device)
        
        # Exploration state
        self.epsilon = self.config.epsilon_start
        self.training = True
        
        # Action space info (for random sampling)
        self.action_dims = self.config.model_config.action_dims
        self.use_continuous = self.config.model_config.use_continuous
    
    def act(
        self,
        observation: Dict[str, np.ndarray],
        deterministic: bool = False
    ) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        Select an action given the current observation.
        
        Uses epsilon-greedy exploration during training.
        
        Args:
            observation: Current observation from environment
            deterministic: If True, always select best action
            
        Returns:
            action: Selected action (as numpy arrays)
            log_prob: Log probability of action
            value: Estimated state value
        """
        # Epsilon-greedy exploration
        if self.training and not deterministic and np.random.random() < self.epsilon:
            return self._random_action(observation)
        
        # Convert observation to tensors
        obs_tensors = self._numpy_to_tensor(observation)
        
        # Use model to select action
        with torch.no_grad():
            action, log_prob, value = self.model.get_action(
                obs_tensors, deterministic=deterministic
            )
        
        # Convert action back to numpy
        action_np = self._action_to_numpy(action)
        log_prob_np = log_prob.cpu().numpy()
        value_np = value.cpu().numpy()
        
        return action_np, log_prob_np, value_np
    
    def _numpy_to_tensor(self, observation: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert numpy observation to tensor."""
        return {
            key: torch.tensor(value, dtype=torch.float32, device=self.device)
            for key, value in observation.items()
        }
    
    def _action_to_numpy(self, action: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Convert tensor action to numpy."""
        if isinstance(action, dict):
            return {
                key: value.cpu().numpy()
                for key, value in action.items()
            }
        return action.cpu().numpy()
    
    def _random_action(
        self,
        observation: Dict[str, np.ndarray]
    ) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """Sample a random action for exploration."""
        # Determine batch size
        blocks_shape = observation['blocks'].shape
        batch_size = blocks_shape[0] if len(blocks_shape) == 4 else 1
        
        if self.use_continuous:
            # Random continuous action
            action = np.random.uniform(
                -1, 1, 
                (batch_size, self.config.model_config.continuous_dim)
            ).astype(np.float32)
            log_prob = np.zeros(batch_size, dtype=np.float32)
        else:
            # Random discrete action
            action = {}
            for action_type, dim in self.action_dims.items():
                action[action_type] = np.random.randint(0, dim, batch_size)
            log_prob = np.zeros(batch_size, dtype=np.float32)
        
        # Get value estimate from model (useful for training)
        obs_tensors = self._numpy_to_tensor(observation)
        with torch.no_grad():
            _, value = self.model.forward(obs_tensors)
        
        value_np = value.squeeze(-1).cpu().numpy()
        
        return action, log_prob, value_np
    
    def evaluate_actions(
        self,
        observations: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate actions for a batch of observations.
        
        Used during training to compute policy loss.
        
        Args:
            observations: Batch of observations
            actions: Batch of actions taken
            
        Returns:
            log_probs: Log probabilities of actions
            values: State value estimates
            entropy: Policy entropy
        """
        # Convert to tensors
        obs_tensors = self._numpy_to_tensor(observations)
        
        if isinstance(actions, dict):
            action_tensors = {
                key: torch.tensor(value, dtype=torch.long, device=self.device)
                for key, value in actions.items()
            }
        else:
            action_tensors = torch.tensor(actions, dtype=torch.float32, device=self.device)
        
        # Evaluate
        log_probs, values, entropy = self.model.evaluate_actions(
            obs_tensors, action_tensors
        )
        
        return (
            log_probs.detach().cpu().numpy(),
            values.detach().cpu().numpy(),
            entropy.detach().cpu().numpy()
        )
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
    
    def set_training(self, training: bool) -> None:
        """Set training mode."""
        self.training = training
        if training:
            self.model.train()
        else:
            self.model.eval()
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get all model parameters."""
        return dict(self.model.named_parameters())
    
    def save(self, path: str) -> None:
        """Save policy to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epsilon': self.epsilon,
            'config': {
                'use_continuous': self.config.model_config.use_continuous,
                'action_dims': self.config.model_config.action_dims,
                'hidden_dim': self.config.model_config.hidden_dim,
                'block_channels': self.config.model_config.block_channels
            }
        }, path)
    
    def load(self, path: str) -> None:
        """Load policy from file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']


class RolloutBuffer:
    """
    Rollout buffer for on-policy algorithms (like PPO/A2C).
    
    Stores complete episodes for batch training.
    Computes advantages and returns.
    """
    
    def __init__(self):
        """Initialize rollout buffer."""
        self.observations: List[Dict] = []
        self.actions: List[Dict] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []
    
    def add(
        self,
        observation: Dict[str, np.ndarray],
        action: Dict,
        reward: float,
        log_prob: float,
        value: float,
        done: bool
    ) -> None:
        """Add a transition to the rollout."""
        self.observations.append({k: v.copy() for k, v in observation.items()})
        self.actions.append(
            {k: v.copy() if isinstance(v, np.ndarray) else v 
             for k, v in action.items()} if isinstance(action, dict) else action
        )
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_returns_and_advantages(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        last_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute returns and GAE advantages.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            last_value: Value estimate for final state
            
        Returns:
            returns: Discounted returns
            advantages: GAE advantages
        """
        n = len(self.rewards)
        returns = np.zeros(n, dtype=np.float32)
        advantages = np.zeros(n, dtype=np.float32)
        
        last_gae = 0.0
        
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])
            
            # TD error
            delta = (
                self.rewards[t] + 
                gamma * next_value * next_non_terminal - 
                self.values[t]
            )
            
            # GAE
            advantages[t] = last_gae = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae
            )
            
            # Returns
            returns[t] = advantages[t] + self.values[t]
        
        return returns, advantages
    
    def get_batch(self) -> Dict:
        """
        Get all data as a batch.
        
        Returns:
            Dictionary with stacked arrays
        """
        batch = {
            'observations': {
                'blocks': np.stack([o['blocks'] for o in self.observations]),
                'inventory': np.stack([o['inventory'] for o in self.observations]),
                'agent_state': np.stack([o['agent_state'] for o in self.observations])
            },
            'rewards': np.array(self.rewards, dtype=np.float32),
            'log_probs': np.array(self.log_probs, dtype=np.float32),
            'values': np.array(self.values, dtype=np.float32),
            'dones': np.array(self.dones, dtype=np.float32)
        }
        
        # Handle actions based on type
        if self.actions and isinstance(self.actions[0], dict):
            batch['actions'] = {}
            for key in self.actions[0]:
                batch['actions'][key] = np.array([a[key] for a in self.actions])
        else:
            batch['actions'] = np.stack(self.actions)
        
        return batch
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
    
    def __len__(self) -> int:
        return len(self.rewards)
