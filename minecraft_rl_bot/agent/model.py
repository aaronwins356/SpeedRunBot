"""
model.py - Neural network architectures for Minecraft RL agent.

This module provides lightweight neural network models that:
- Process 3D block observations
- Handle auxiliary inputs (inventory, agent state)
- Output action distributions

The models are designed to be:
- Efficient for low-resource hardware (no GPU required)
- Modular for easy modification
- Compatible with both discrete and continuous action spaces

Model Components:
1. Block Encoder: 3D CNN for processing block observations
2. State Encoder: MLP for processing auxiliary state
3. Policy Head: Output layer for action probabilities/values
4. Value Head: Output layer for state value estimation
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

# Use numpy for core operations to keep things lightweight
# PyTorch import is optional and wrapped for CPU-only use


@dataclass
class ModelConfig:
    """
    Configuration for the neural network model.
    
    Attributes:
        block_channels: Number of output channels from block encoder
        hidden_dim: Dimension of hidden layers
        num_block_types: Number of distinct block types
        obs_size: Size of observation region (21x21x21)
        inventory_size: Number of inventory item types
        state_size: Size of agent state vector
        action_dims: Dictionary of action space dimensions
        use_continuous: Whether to use continuous action space
    """
    block_channels: int = 32
    hidden_dim: int = 128
    num_block_types: int = 64
    obs_size: int = 21
    inventory_size: int = 10
    state_size: int = 13
    action_dims: Dict[str, int] = None
    use_continuous: bool = False
    continuous_dim: int = 7
    
    def __post_init__(self):
        if self.action_dims is None:
            self.action_dims = {
                'movement': 8,
                'camera': 5,
                'interaction': 7,
                'inventory': 12
            }


class BlockEncoder:
    """
    3D Convolutional encoder for block observations.
    
    Processes a 21x21x21 block tensor and outputs a feature vector.
    Uses simple convolutions that can run efficiently on CPU.
    
    Architecture:
    - Embedding layer for block types
    - 3D convolutions with stride to reduce spatial dimensions
    - Global average pooling
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
        # Block type embedding (one-hot or learned)
        self.num_types = config.num_block_types
        
        # Convolution weights (initialized randomly, will be trained)
        # Using numpy arrays for weights
        self.conv1_weights = self._init_conv3d(1, 16, 3)  # 21 -> 19
        self.conv2_weights = self._init_conv3d(16, 32, 3, stride=2)  # 19 -> 9
        self.conv3_weights = self._init_conv3d(32, config.block_channels, 3, stride=2)  # 9 -> 4
        
        self.conv1_bias = np.zeros(16)
        self.conv2_bias = np.zeros(32)
        self.conv3_bias = np.zeros(config.block_channels)
    
    def _init_conv3d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1
    ) -> np.ndarray:
        """Initialize 3D convolution weights with Xavier initialization."""
        fan_in = in_channels * kernel_size ** 3
        fan_out = out_channels * kernel_size ** 3
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size) * std
    
    def forward(self, blocks: np.ndarray) -> np.ndarray:
        """
        Forward pass through the block encoder.
        
        Args:
            blocks: Block observation tensor of shape (batch, 21, 21, 21)
                   or (21, 21, 21) with integer block type values
                   
        Returns:
            Feature vector of shape (batch, block_channels)
        """
        # Handle input shape: expect (batch, d, h, w) or (d, h, w)
        if len(blocks.shape) == 3:
            blocks = blocks[np.newaxis, ...]  # Add batch dimension
        
        batch_size = blocks.shape[0]
        
        # Convert block types to one-hot (simplified: use normalized values)
        x = blocks.astype(np.float32) / self.num_types
        x = x[:, np.newaxis, ...]  # Add channel dimension: (batch, 1, d, h, w)
        
        # Apply convolutions with ReLU
        x = self._conv3d(x, self.conv1_weights, self.conv1_bias)
        x = np.maximum(x, 0)  # ReLU
        
        x = self._conv3d(x, self.conv2_weights, self.conv2_bias, stride=2)
        x = np.maximum(x, 0)
        
        x = self._conv3d(x, self.conv3_weights, self.conv3_bias, stride=2)
        x = np.maximum(x, 0)
        
        # Global average pooling
        x = x.mean(axis=(2, 3, 4))
        
        return x
    
    def _conv3d(
        self,
        x: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray,
        stride: int = 1
    ) -> np.ndarray:
        """
        Simple 3D convolution implementation.
        
        This is a basic implementation for CPU inference.
        For training, consider using PyTorch or JAX.
        """
        batch, in_channels, d, h, w = x.shape
        out_channels, _, kd, kh, kw = weights.shape
        
        # Output dimensions
        out_d = (d - kd) // stride + 1
        out_h = (h - kh) // stride + 1
        out_w = (w - kw) // stride + 1
        
        output = np.zeros((batch, out_channels, out_d, out_h, out_w))
        
        for b in range(batch):
            for oc in range(out_channels):
                for od in range(out_d):
                    for oh in range(out_h):
                        for ow in range(out_w):
                            id_start = od * stride
                            ih_start = oh * stride
                            iw_start = ow * stride
                            
                            patch = x[b, :, 
                                      id_start:id_start+kd,
                                      ih_start:ih_start+kh,
                                      iw_start:iw_start+kw]
                            
                            output[b, oc, od, oh, ow] = np.sum(
                                patch * weights[oc]
                            ) + bias[oc]
        
        return output
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get all trainable parameters."""
        return {
            'conv1_weights': self.conv1_weights,
            'conv1_bias': self.conv1_bias,
            'conv2_weights': self.conv2_weights,
            'conv2_bias': self.conv2_bias,
            'conv3_weights': self.conv3_weights,
            'conv3_bias': self.conv3_bias
        }
    
    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Set trainable parameters."""
        self.conv1_weights = params['conv1_weights']
        self.conv1_bias = params['conv1_bias']
        self.conv2_weights = params['conv2_weights']
        self.conv2_bias = params['conv2_bias']
        self.conv3_weights = params['conv3_weights']
        self.conv3_bias = params['conv3_bias']


class StateEncoder:
    """
    MLP encoder for inventory and agent state.
    
    Processes auxiliary inputs and outputs a feature vector
    that is combined with the block features.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
        input_dim = config.inventory_size + config.state_size
        hidden_dim = config.hidden_dim // 2
        
        # MLP weights
        self.fc1_weights = self._init_linear(input_dim, hidden_dim)
        self.fc1_bias = np.zeros(hidden_dim)
        
        self.fc2_weights = self._init_linear(hidden_dim, hidden_dim)
        self.fc2_bias = np.zeros(hidden_dim)
    
    def _init_linear(self, in_features: int, out_features: int) -> np.ndarray:
        """Initialize linear layer weights."""
        std = np.sqrt(2.0 / (in_features + out_features))
        return np.random.randn(out_features, in_features) * std
    
    def forward(
        self,
        inventory: np.ndarray,
        agent_state: np.ndarray
    ) -> np.ndarray:
        """
        Forward pass through the state encoder.
        
        Args:
            inventory: Inventory counts of shape (batch, inventory_size)
            agent_state: Agent state of shape (batch, state_size)
            
        Returns:
            Feature vector of shape (batch, hidden_dim // 2)
        """
        # Ensure batch dimension
        if len(inventory.shape) == 1:
            inventory = inventory[np.newaxis, :]
            agent_state = agent_state[np.newaxis, :]
        
        # Concatenate inputs
        x = np.concatenate([inventory, agent_state], axis=-1)
        
        # MLP with ReLU
        x = np.dot(x, self.fc1_weights.T) + self.fc1_bias
        x = np.maximum(x, 0)
        
        x = np.dot(x, self.fc2_weights.T) + self.fc2_bias
        x = np.maximum(x, 0)
        
        return x
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get all trainable parameters."""
        return {
            'fc1_weights': self.fc1_weights,
            'fc1_bias': self.fc1_bias,
            'fc2_weights': self.fc2_weights,
            'fc2_bias': self.fc2_bias
        }
    
    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Set trainable parameters."""
        self.fc1_weights = params['fc1_weights']
        self.fc1_bias = params['fc1_bias']
        self.fc2_weights = params['fc2_weights']
        self.fc2_bias = params['fc2_bias']


class MinecraftPolicy:
    """
    Complete policy network for Minecraft RL agent.
    
    Combines block encoder and state encoder, then outputs:
    - Action logits for discrete actions
    - Action parameters for continuous actions
    - Value estimate for the current state
    
    This is an actor-critic architecture where:
    - Actor: Outputs action distribution
    - Critic: Outputs state value estimate
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
        # Encoders
        self.block_encoder = BlockEncoder(config)
        self.state_encoder = StateEncoder(config)
        
        # Combined feature dimension
        combined_dim = config.block_channels + config.hidden_dim // 2
        
        # Shared hidden layer
        self.shared_weights = self._init_linear(combined_dim, config.hidden_dim)
        self.shared_bias = np.zeros(config.hidden_dim)
        
        # Policy heads (one for each action type in discrete mode)
        if not config.use_continuous:
            self.policy_heads = {}
            for action_type, dim in config.action_dims.items():
                self.policy_heads[action_type] = {
                    'weights': self._init_linear(config.hidden_dim, dim),
                    'bias': np.zeros(dim)
                }
        else:
            # Continuous action head (mean and log_std)
            self.action_mean_weights = self._init_linear(
                config.hidden_dim, config.continuous_dim
            )
            self.action_mean_bias = np.zeros(config.continuous_dim)
            self.action_log_std = np.zeros(config.continuous_dim)
        
        # Value head
        self.value_weights = self._init_linear(config.hidden_dim, 1)
        self.value_bias = np.zeros(1)
    
    def _init_linear(self, in_features: int, out_features: int) -> np.ndarray:
        """Initialize linear layer weights."""
        std = np.sqrt(2.0 / (in_features + out_features))
        return np.random.randn(out_features, in_features) * std
    
    def forward(
        self,
        observation: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Forward pass through the policy network.
        
        Args:
            observation: Dictionary containing:
                - blocks: Block observation (batch, 21, 21, 21)
                - inventory: Inventory counts (batch, inventory_size)
                - agent_state: Agent state (batch, state_size)
                
        Returns:
            action_output: Either action logits (discrete) or 
                          mean/std (continuous)
            value: State value estimate (batch, 1)
        """
        # Encode observations
        block_features = self.block_encoder.forward(observation['blocks'])
        state_features = self.state_encoder.forward(
            observation['inventory'],
            observation['agent_state']
        )
        
        # Combine features
        combined = np.concatenate([block_features, state_features], axis=-1)
        
        # Shared hidden layer
        hidden = np.dot(combined, self.shared_weights.T) + self.shared_bias
        hidden = np.maximum(hidden, 0)  # ReLU
        
        # Compute outputs
        if not self.config.use_continuous:
            # Discrete action logits
            action_output = {}
            for action_type, head in self.policy_heads.items():
                logits = np.dot(hidden, head['weights'].T) + head['bias']
                action_output[action_type] = logits
        else:
            # Continuous action parameters
            mean = np.dot(hidden, self.action_mean_weights.T) + self.action_mean_bias
            log_std = self.action_log_std
            action_output = {'mean': mean, 'log_std': log_std}
        
        # Value estimate
        value = np.dot(hidden, self.value_weights.T) + self.value_bias
        
        return action_output, value
    
    def get_action(
        self,
        observation: Dict[str, np.ndarray],
        deterministic: bool = False
    ) -> Tuple[Dict | np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample an action from the policy.
        
        Args:
            observation: Current observation
            deterministic: If True, return most likely action
            
        Returns:
            action: Sampled or deterministic action
            log_prob: Log probability of the action
            value: State value estimate
        """
        action_output, value = self.forward(observation)
        
        if not self.config.use_continuous:
            # Sample discrete actions
            action = {}
            log_probs = []
            
            for action_type, logits in action_output.items():
                # Softmax
                probs = self._softmax(logits)
                
                if deterministic:
                    action_idx = np.argmax(probs, axis=-1)
                else:
                    # Sample from categorical distribution
                    action_idx = np.array([
                        np.random.choice(len(p), p=p) for p in probs
                    ])
                
                action[action_type] = action_idx
                
                # Log probability
                log_prob = np.log(probs[np.arange(len(probs)), action_idx] + 1e-8)
                log_probs.append(log_prob)
            
            total_log_prob = np.sum(log_probs, axis=0)
            
        else:
            # Sample continuous actions
            mean = action_output['mean']
            std = np.exp(action_output['log_std'])
            
            if deterministic:
                action = mean
            else:
                action = mean + std * np.random.randn(*mean.shape)
            
            # Log probability (Gaussian)
            log_prob = -0.5 * np.sum(
                ((action - mean) / std) ** 2 + 
                2 * action_output['log_std'] + 
                np.log(2 * np.pi),
                axis=-1
            )
            total_log_prob = log_prob
        
        return action, total_log_prob, value.squeeze(-1)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax along last axis."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get all trainable parameters."""
        params = {}
        
        # Encoder parameters
        for name, value in self.block_encoder.get_parameters().items():
            params[f'block_encoder.{name}'] = value
        for name, value in self.state_encoder.get_parameters().items():
            params[f'state_encoder.{name}'] = value
        
        # Shared layer
        params['shared_weights'] = self.shared_weights
        params['shared_bias'] = self.shared_bias
        
        # Policy heads
        if not self.config.use_continuous:
            for action_type, head in self.policy_heads.items():
                params[f'policy.{action_type}.weights'] = head['weights']
                params[f'policy.{action_type}.bias'] = head['bias']
        else:
            params['action_mean_weights'] = self.action_mean_weights
            params['action_mean_bias'] = self.action_mean_bias
            params['action_log_std'] = self.action_log_std
        
        # Value head
        params['value_weights'] = self.value_weights
        params['value_bias'] = self.value_bias
        
        return params
    
    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Set all trainable parameters."""
        # Encoder parameters
        block_params = {
            k.replace('block_encoder.', ''): v
            for k, v in params.items()
            if k.startswith('block_encoder.')
        }
        self.block_encoder.set_parameters(block_params)
        
        state_params = {
            k.replace('state_encoder.', ''): v
            for k, v in params.items()
            if k.startswith('state_encoder.')
        }
        self.state_encoder.set_parameters(state_params)
        
        # Shared layer
        self.shared_weights = params['shared_weights']
        self.shared_bias = params['shared_bias']
        
        # Policy heads
        if not self.config.use_continuous:
            for action_type in self.policy_heads:
                self.policy_heads[action_type]['weights'] = params[f'policy.{action_type}.weights']
                self.policy_heads[action_type]['bias'] = params[f'policy.{action_type}.bias']
        else:
            self.action_mean_weights = params['action_mean_weights']
            self.action_mean_bias = params['action_mean_bias']
            self.action_log_std = params['action_log_std']
        
        # Value head
        self.value_weights = params['value_weights']
        self.value_bias = params['value_bias']
    
    def save(self, path: str) -> None:
        """Save model parameters to file."""
        params = self.get_parameters()
        np.savez(path, **params)
    
    def load(self, path: str) -> None:
        """Load model parameters from file."""
        data = np.load(path)
        params = {key: data[key] for key in data.files}
        self.set_parameters(params)


def create_model(config: Optional[ModelConfig] = None) -> MinecraftPolicy:
    """
    Factory function to create a MinecraftPolicy model.
    
    Args:
        config: Model configuration (uses defaults if None)
        
    Returns:
        Initialized MinecraftPolicy model
    """
    if config is None:
        config = ModelConfig()
    return MinecraftPolicy(config)
