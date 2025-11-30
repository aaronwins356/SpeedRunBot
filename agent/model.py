"""
model.py - PyTorch neural network architectures for Minecraft RL agent.

This module provides lightweight neural network models that:
- Process 3D block observations (21x21x21 tensor)
- Handle auxiliary inputs (inventory, agent state)
- Output action distributions for discrete/continuous actions

The models are designed to be:
- Efficient for low-resource hardware (CPU-friendly)
- Modular for easy modification
- Compatible with both discrete and continuous action spaces

Model Components:
1. Block Encoder: 3D CNN for processing block observations
2. State Encoder: MLP for processing auxiliary state
3. Policy Head: Output layer for action probabilities
4. Value Head: Output layer for state value estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


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
        continuous_dim: Dimension of continuous action space
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


class BlockEncoder(nn.Module):
    """
    3D Convolutional encoder for block observations.
    
    Processes a 21x21x21 block tensor and outputs a feature vector.
    Uses efficient 3D convolutions suitable for CPU training.
    
    Architecture:
    - Embedding layer for block types
    - 3D convolutions with stride to reduce spatial dimensions
    - Global average pooling to feature vector
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Block type embedding (learnable)
        self.block_embedding = nn.Embedding(
            config.num_block_types, 
            8  # Embedding dimension
        )
        
        # 3D Convolutional layers
        # Input: (batch, 8, 21, 21, 21)
        self.conv1 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=0)  # -> (batch, 16, 19, 19, 19)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=0)  # -> (batch, 32, 9, 9, 9)
        self.conv3 = nn.Conv3d(32, config.block_channels, kernel_size=3, stride=2, padding=0)  # -> (batch, 32, 4, 4, 4)
        
        # Batch normalization for stable training
        self.bn1 = nn.BatchNorm3d(16)
        self.bn2 = nn.BatchNorm3d(32)
        self.bn3 = nn.BatchNorm3d(config.block_channels)
    
    def forward(self, blocks: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the block encoder.
        
        Args:
            blocks: Block observation tensor of shape (batch, 21, 21, 21)
                   with integer block type values
                   
        Returns:
            Feature vector of shape (batch, block_channels)
        """
        # Ensure input is long tensor for embedding
        blocks = blocks.long()
        
        # Clamp block types to valid range
        blocks = torch.clamp(blocks, 0, self.config.num_block_types - 1)
        
        # Get block embeddings: (batch, 21, 21, 21) -> (batch, 21, 21, 21, 8)
        x = self.block_embedding(blocks)
        
        # Permute to channel-first format: (batch, 8, 21, 21, 21)
        x = x.permute(0, 4, 1, 2, 3)
        
        # Apply convolutions with ReLU and batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global average pooling: (batch, channels, d, h, w) -> (batch, channels)
        x = x.mean(dim=[2, 3, 4])
        
        return x


class StateEncoder(nn.Module):
    """
    MLP encoder for inventory and agent state.
    
    Processes auxiliary inputs and outputs a feature vector
    that is combined with the block features.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        input_dim = config.inventory_size + config.state_size
        hidden_dim = config.hidden_dim // 2
        
        # MLP layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization for stable training
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        inventory: torch.Tensor,
        agent_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the state encoder.
        
        Args:
            inventory: Inventory counts of shape (batch, inventory_size)
            agent_state: Agent state of shape (batch, state_size)
            
        Returns:
            Feature vector of shape (batch, hidden_dim // 2)
        """
        # Concatenate inputs
        x = torch.cat([inventory, agent_state], dim=-1)
        
        # MLP with ReLU and layer norm
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        
        return x


class MinecraftPolicy(nn.Module):
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
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        
        if config is None:
            config = ModelConfig()
        self.config = config
        
        # Encoders
        self.block_encoder = BlockEncoder(config)
        self.state_encoder = StateEncoder(config)
        
        # Combined feature dimension
        combined_dim = config.block_channels + config.hidden_dim // 2
        
        # Shared hidden layer
        self.shared_fc = nn.Linear(combined_dim, config.hidden_dim)
        self.shared_ln = nn.LayerNorm(config.hidden_dim)
        
        # Policy heads (one for each action type in discrete mode)
        if not config.use_continuous:
            self.policy_heads = nn.ModuleDict({
                action_type: nn.Linear(config.hidden_dim, dim)
                for action_type, dim in config.action_dims.items()
            })
        else:
            # Continuous action head (mean and log_std)
            self.action_mean = nn.Linear(config.hidden_dim, config.continuous_dim)
            self.action_log_std = nn.Parameter(torch.zeros(config.continuous_dim))
        
        # Value head
        self.value_fc = nn.Linear(config.hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv3d)):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        observation: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            observation: Dictionary containing:
                - blocks: Block observation (batch, 21, 21, 21)
                - inventory: Inventory counts (batch, inventory_size)
                - agent_state: Agent state (batch, state_size)
                
        Returns:
            action_output: Either action logits (discrete) or mean/log_std (continuous)
            value: State value estimate (batch, 1)
        """
        # Encode observations
        block_features = self.block_encoder(observation['blocks'])
        state_features = self.state_encoder(
            observation['inventory'],
            observation['agent_state']
        )
        
        # Combine features
        combined = torch.cat([block_features, state_features], dim=-1)
        
        # Shared hidden layer
        hidden = F.relu(self.shared_ln(self.shared_fc(combined)))
        
        # Compute outputs
        if not self.config.use_continuous:
            # Discrete action logits
            action_output = {
                action_type: head(hidden)
                for action_type, head in self.policy_heads.items()
            }
        else:
            # Continuous action parameters
            mean = self.action_mean(hidden)
            action_output = {
                'mean': mean,
                'log_std': self.action_log_std.expand_as(mean)
            }
        
        # Value estimate
        value = self.value_fc(hidden)
        
        return action_output, value
    
    def get_action(
        self,
        observation: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
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
                probs = F.softmax(logits, dim=-1)
                
                if deterministic:
                    action_idx = torch.argmax(probs, dim=-1)
                else:
                    # Sample from categorical distribution
                    dist = torch.distributions.Categorical(probs)
                    action_idx = dist.sample()
                
                action[action_type] = action_idx
                
                # Log probability
                log_prob = F.log_softmax(logits, dim=-1)
                log_prob = log_prob.gather(-1, action_idx.unsqueeze(-1)).squeeze(-1)
                log_probs.append(log_prob)
            
            total_log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)
            
        else:
            # Sample continuous actions
            mean = action_output['mean']
            std = torch.exp(action_output['log_std'])
            
            if deterministic:
                action = mean
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
            
            # Log probability (Gaussian)
            log_prob = -0.5 * (
                ((action - mean) / std) ** 2 +
                2 * action_output['log_std'] +
                torch.log(torch.tensor(2 * 3.14159))
            )
            total_log_prob = log_prob.sum(dim=-1)
        
        return action, total_log_prob, value.squeeze(-1)
    
    def evaluate_actions(
        self,
        observations: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        action_output, values = self.forward(observations)
        
        if not self.config.use_continuous:
            # Discrete actions
            log_probs = []
            entropies = []
            
            for action_type, logits in action_output.items():
                probs = F.softmax(logits, dim=-1)
                log_prob_all = F.log_softmax(logits, dim=-1)
                
                # Handle action indexing
                if action_type in actions:
                    action_indices = actions[action_type]
                    # Ensure action_indices is at least 1D
                    if action_indices.dim() == 0:
                        action_indices = action_indices.unsqueeze(0)
                    # Ensure action_indices matches logits batch dim
                    if action_indices.dim() == 1 and logits.dim() == 2:
                        # action_indices: (batch,) -> need (batch, 1) for gather
                        gather_indices = action_indices.unsqueeze(-1)
                    else:
                        gather_indices = action_indices
                    
                    # Log probability of taken actions
                    log_prob = log_prob_all.gather(-1, gather_indices).squeeze(-1)
                    log_probs.append(log_prob)
                    
                    # Entropy
                    entropy = -torch.sum(probs * log_prob_all, dim=-1)
                    entropies.append(entropy)
            
            total_log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)
            total_entropy = torch.stack(entropies, dim=-1).sum(dim=-1)
        else:
            # Continuous actions
            mean = action_output['mean']
            log_std = action_output['log_std']
            std = torch.exp(log_std)
            
            # Gaussian log probability
            log_prob = -0.5 * (
                ((actions - mean) / std) ** 2 +
                2 * log_std +
                torch.log(torch.tensor(2 * 3.14159))
            )
            total_log_prob = log_prob.sum(dim=-1)
            
            # Gaussian entropy
            total_entropy = 0.5 * (
                1 + torch.log(torch.tensor(2 * 3.14159)) + 2 * log_std
            ).sum(dim=-1)
        
        return total_log_prob, values.squeeze(-1), total_entropy


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
