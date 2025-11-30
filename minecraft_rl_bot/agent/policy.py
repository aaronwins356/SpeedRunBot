"""
policy.py - Policy wrapper and action selection for Minecraft RL agent.

This module provides high-level policy interfaces that:
- Wrap the neural network model
- Handle action selection and exploration
- Manage policy updates during training

The policy is the main interface between the training loop and the model.
"""

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
        
        # Create model
        self.model = create_model(self.config.model_config)
        
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
    ) -> Tuple[Dict | np.ndarray, np.ndarray, np.ndarray]:
        """
        Select an action given the current observation.
        
        Uses epsilon-greedy exploration during training.
        
        Args:
            observation: Current observation from environment
            deterministic: If True, always select best action
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: Estimated state value
        """
        # Epsilon-greedy exploration
        if self.training and not deterministic and np.random.random() < self.epsilon:
            return self._random_action(observation)
        
        # Use model to select action
        return self.model.get_action(observation, deterministic=deterministic)
    
    def _random_action(
        self,
        observation: Dict[str, np.ndarray]
    ) -> Tuple[Dict | np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random action for exploration."""
        batch_size = observation['blocks'].shape[0] if len(observation['blocks'].shape) == 4 else 1
        
        if self.use_continuous:
            # Random continuous action
            action = np.random.uniform(-1, 1, (batch_size, self.config.model_config.continuous_dim))
            log_prob = np.zeros(batch_size)
        else:
            # Random discrete action
            action = {}
            for action_type, dim in self.action_dims.items():
                action[action_type] = np.random.randint(0, dim, batch_size)
            log_prob = np.zeros(batch_size)
        
        # Get value estimate from model (useful for training)
        _, value = self.model.forward(observation)
        
        return action, log_prob, value.squeeze(-1)
    
    def evaluate_actions(
        self,
        observations: Dict[str, np.ndarray],
        actions: Dict | np.ndarray
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
        action_output, values = self.model.forward(observations)
        
        if not self.use_continuous:
            # Discrete actions
            log_probs = []
            entropies = []
            
            for action_type, logits in action_output.items():
                probs = self._softmax(logits)
                action_indices = actions[action_type]
                
                # Log probability of taken actions
                log_prob = np.log(
                    probs[np.arange(len(probs)), action_indices] + 1e-8
                )
                log_probs.append(log_prob)
                
                # Entropy
                entropy = -np.sum(probs * np.log(probs + 1e-8), axis=-1)
                entropies.append(entropy)
            
            total_log_prob = np.sum(log_probs, axis=0)
            total_entropy = np.sum(entropies, axis=0)
        else:
            # Continuous actions
            mean = action_output['mean']
            log_std = action_output['log_std']
            std = np.exp(log_std)
            
            # Gaussian log probability
            total_log_prob = -0.5 * np.sum(
                ((actions - mean) / std) ** 2 +
                2 * log_std +
                np.log(2 * np.pi),
                axis=-1
            )
            
            # Gaussian entropy
            total_entropy = 0.5 * np.sum(
                1 + np.log(2 * np.pi) + 2 * log_std,
                axis=-1
            )
        
        return total_log_prob, values.squeeze(-1), total_entropy
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax along last axis."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
    
    def set_training(self, training: bool) -> None:
        """Set training mode."""
        self.training = training
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get all model parameters."""
        return self.model.get_parameters()
    
    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Set all model parameters."""
        self.model.set_parameters(params)
    
    def save(self, path: str) -> None:
        """Save policy to file."""
        self.model.save(path)
    
    def load(self, path: str) -> None:
        """Load policy from file."""
        self.model.load(path)


class ReplayBuffer:
    """
    Simple replay buffer for storing experience.
    
    Stores transitions (s, a, r, s', done) for training.
    Supports random sampling for batch training.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer: List[Dict] = []
        self.position = 0
    
    def push(
        self,
        observation: Dict[str, np.ndarray],
        action: Dict | np.ndarray,
        reward: float,
        next_observation: Dict[str, np.ndarray],
        done: bool,
        log_prob: float = 0.0,
        value: float = 0.0
    ) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            observation: State before action
            action: Action taken
            reward: Reward received
            next_observation: State after action
            done: Whether episode ended
            log_prob: Log probability of action
            value: Estimated value of state
        """
        transition = {
            'observation': {k: v.copy() for k, v in observation.items()},
            'action': action.copy() if isinstance(action, dict) else action,
            'reward': reward,
            'next_observation': {k: v.copy() for k, v in next_observation.items()},
            'done': done,
            'log_prob': log_prob,
            'value': value
        }
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Dict:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Batch dictionary with stacked arrays
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        batch = {
            'observations': {
                'blocks': [],
                'inventory': [],
                'agent_state': []
            },
            'actions': {'movement': [], 'camera': [], 'interaction': [], 'inventory': []},
            'rewards': [],
            'next_observations': {
                'blocks': [],
                'inventory': [],
                'agent_state': []
            },
            'dones': [],
            'log_probs': [],
            'values': []
        }
        
        for idx in indices:
            t = self.buffer[idx]
            
            for key in batch['observations']:
                batch['observations'][key].append(t['observation'][key])
                batch['next_observations'][key].append(t['next_observation'][key])
            
            if isinstance(t['action'], dict):
                for key in batch['actions']:
                    if key in t['action']:
                        batch['actions'][key].append(t['action'][key])
            
            batch['rewards'].append(t['reward'])
            batch['dones'].append(t['done'])
            batch['log_probs'].append(t['log_prob'])
            batch['values'].append(t['value'])
        
        # Stack arrays
        for key in batch['observations']:
            batch['observations'][key] = np.stack(batch['observations'][key])
            batch['next_observations'][key] = np.stack(batch['next_observations'][key])
        
        for key in batch['actions']:
            if batch['actions'][key]:
                batch['actions'][key] = np.array(batch['actions'][key])
        
        batch['rewards'] = np.array(batch['rewards'])
        batch['dones'] = np.array(batch['dones'])
        batch['log_probs'] = np.array(batch['log_probs'])
        batch['values'] = np.array(batch['values'])
        
        return batch
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.position = 0
    
    def __len__(self) -> int:
        return len(self.buffer)


class RolloutBuffer:
    """
    Rollout buffer for on-policy algorithms (like PPO).
    
    Stores complete episodes for batch training.
    Computes advantages and returns.
    """
    
    def __init__(self):
        """Initialize rollout buffer."""
        self.observations: List[Dict] = []
        self.actions: List[Dict | np.ndarray] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []
    
    def add(
        self,
        observation: Dict[str, np.ndarray],
        action: Dict | np.ndarray,
        reward: float,
        log_prob: float,
        value: float,
        done: bool
    ) -> None:
        """Add a transition to the rollout."""
        self.observations.append({k: v.copy() for k, v in observation.items()})
        self.actions.append(action.copy() if isinstance(action, dict) else action)
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
        returns = np.zeros(n)
        advantages = np.zeros(n)
        
        last_gae = 0
        
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
            'rewards': np.array(self.rewards),
            'log_probs': np.array(self.log_probs),
            'values': np.array(self.values),
            'dones': np.array(self.dones)
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
