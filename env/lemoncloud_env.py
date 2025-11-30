"""
lemoncloud_env.py - Live environment wrapper for LemonCloud server.

This module provides a Gym-like interface that wraps the live
Minecraft server connection, allowing the trained RL policy to
be used for inference in real gameplay.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass

from integration.mc_client import MinecraftClient, ClientConfig, Position

logger = logging.getLogger(__name__)


@dataclass
class LiveEnvConfig:
    """Configuration for the live environment."""
    # Client settings
    host: str = "play.lemoncloud.net"
    port: int = 25565
    username: str = ""
    
    # Observation settings
    obs_size: int = 21
    inventory_size: int = 10
    state_size: int = 13
    
    # Action settings
    action_dims: Dict[str, int] = None
    
    # Environment limits
    max_steps: int = 100000
    
    def __post_init__(self):
        if self.action_dims is None:
            self.action_dims = {
                'movement': 8,
                'camera': 5,
                'interaction': 7,
                'inventory': 12
            }


class LemonCloudEnv:
    """
    Gym-like environment wrapper for live Minecraft gameplay.
    
    This class provides the same interface as the training environment,
    allowing the trained RL policy to be used for inference on a
    live server.
    
    Interface:
    - reset(): Initialize/reconnect and return observation
    - step(action): Execute action and return (obs, reward, done, truncated, info)
    - observe(): Get current observation without taking action
    
    Usage:
        env = LemonCloudEnv(config)
        obs = env.reset()
        while True:
            action = policy.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break
    """
    
    # Block ID mapping for observations
    BLOCK_ID_MAP = {
        "minecraft:air": 0,
        "minecraft:stone": 1,
        "minecraft:end_stone": 2,
        "minecraft:end_stone_bricks": 3,
        "minecraft:purpur_block": 4,
        "minecraft:purpur_pillar": 5,
        "minecraft:purpur_stairs": 6,
        "minecraft:end_rod": 7,
        "minecraft:chest": 8,
        "minecraft:dragon_head": 9,
        "minecraft:brewing_stand": 10,
        # Add more as needed
    }
    
    def __init__(
        self,
        config: Optional[LiveEnvConfig] = None,
        client: Optional[MinecraftClient] = None
    ):
        """
        Initialize the live environment.
        
        Args:
            config: Environment configuration
            client: Optional pre-configured client
        """
        self.config = config or LiveEnvConfig()
        
        # Create or use provided client
        if client is not None:
            self.client = client
        else:
            client_config = ClientConfig(
                host=self.config.host,
                port=self.config.port,
                username=self.config.username
            )
            self.client = MinecraftClient(client_config)
        
        # Environment state
        self._step_count = 0
        self._episode_reward = 0.0
        self._prev_health = 20.0
        self._prev_position: Optional[Position] = None
        
        # Observation cache
        self._last_observation: Optional[Dict[str, np.ndarray]] = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset the environment.
        
        For a live environment, this means:
        - Ensuring connection to server
        - Resetting internal tracking state
        
        Args:
            seed: Ignored (live environment)
            options: Optional reset options
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        # Ensure connected
        if not self.client.is_connected():
            logger.info("Connecting to server...")
            if not self.client.connect():
                raise RuntimeError("Failed to connect to server")
        
        # Reset internal state
        self._step_count = 0
        self._episode_reward = 0.0
        self._prev_health = self.client.get_health()
        self._prev_position = self.client.get_position()
        
        # Get initial observation
        observation = self.observe()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self,
        action: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action dictionary matching training format
            
        Returns:
            observation: New observation after action
            reward: Reward for this step
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional information
        """
        self._step_count += 1
        
        # Execute action
        self._execute_action(action)
        
        # Update client state
        self.client.update()
        
        # Get new observation
        observation = self.observe()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        terminated = self._check_terminated()
        truncated = self._step_count >= self.config.max_steps
        
        # Get info
        info = self._get_info()
        
        # Track
        self._episode_reward += reward
        self._prev_health = self.client.get_health()
        self._prev_position = self.client.get_position()
        
        return observation, reward, terminated, truncated, info
    
    def observe(self) -> Dict[str, np.ndarray]:
        """
        Get current observation without taking action.
        
        Returns:
            Observation dictionary with:
            - blocks: (obs_size, obs_size, obs_size) block types
            - inventory: (inventory_size,) item counts
            - agent_state: (state_size,) agent state values
        """
        pos = self.client.get_position()
        if pos is None:
            pos = Position(0, 64, 0)
        
        # Block observation
        blocks = self._get_block_observation(pos)
        
        # Inventory observation
        inventory = self._get_inventory_observation()
        
        # Agent state
        agent_state = self._get_agent_state(pos)
        
        observation = {
            'blocks': blocks,
            'inventory': inventory,
            'agent_state': agent_state
        }
        
        self._last_observation = observation
        return observation
    
    def _get_block_observation(self, center: Position) -> np.ndarray:
        """Get block observation cube."""
        obs_size = self.config.obs_size
        half_size = obs_size // 2
        
        blocks = np.zeros((obs_size, obs_size, obs_size), dtype=np.float32)
        
        cx, cy, cz = int(center.x), int(center.y), int(center.z)
        
        for dx in range(-half_size, half_size + 1):
            for dy in range(-half_size, half_size + 1):
                for dz in range(-half_size, half_size + 1):
                    x, y, z = cx + dx, cy + dy, cz + dz
                    
                    block = self.client.get_block_at(x, y, z)
                    
                    if block:
                        block_id = self.BLOCK_ID_MAP.get(block.block_id, 11)
                    else:
                        block_id = 0
                    
                    bx = dx + half_size
                    by = dy + half_size
                    bz = dz + half_size
                    
                    blocks[bx, by, bz] = block_id
        
        return blocks
    
    def _get_inventory_observation(self) -> np.ndarray:
        """Get inventory observation."""
        inventory = np.zeros(self.config.inventory_size, dtype=np.float32)
        
        items = self.client.get_inventory()
        
        for item in items:
            if item is None:
                continue
            
            # Categorize items
            if "pickaxe" in item.item_id:
                inventory[0] += item.count
            elif "sword" in item.item_id:
                inventory[1] += item.count
            elif "block" in item.item_id:
                inventory[2] += item.count
            elif "food" in item.item_id:
                inventory[3] += item.count
            elif "elytra" in item.item_id:
                inventory[4] += item.count
        
        # Normalize
        inventory = np.clip(inventory / 64.0, 0, 1)
        
        return inventory
    
    def _get_agent_state(self, pos: Position) -> np.ndarray:
        """Get agent state observation."""
        state = np.zeros(self.config.state_size, dtype=np.float32)
        
        # Position
        state[0] = pos.x / 100.0
        state[1] = pos.y / 256.0
        state[2] = pos.z / 100.0
        
        # Velocity (placeholder)
        state[3] = 0.0
        state[4] = 0.0
        state[5] = 0.0
        
        # Rotation (placeholder)
        state[6] = 0.0
        state[7] = 0.0
        
        # Health and food
        state[8] = self.client.get_health() / 20.0
        state[9] = self.client.get_food() / 20.0
        
        # Dimension flags
        dimension = self.client.get_dimension()
        state[10] = 1.0 if dimension == "minecraft:the_nether" else 0.0
        state[11] = 1.0 if dimension == "minecraft:the_end" else 0.0
        
        # Day count (placeholder)
        state[12] = 0.0
        
        return state
    
    def _execute_action(self, action: Dict[str, np.ndarray]) -> None:
        """Execute action on the client."""
        # Movement action
        movement = int(action.get('movement', [0])[0])
        self._execute_movement(movement)
        
        # Camera action
        camera = int(action.get('camera', [0])[0])
        self._execute_camera(camera)
        
        # Interaction action
        interaction = int(action.get('interaction', [0])[0])
        self._execute_interaction(interaction)
    
    def _execute_movement(self, movement: int) -> None:
        """Execute movement action."""
        if movement == 0:
            pass  # No movement
        elif movement == 1:
            self.client.move(forward=1.0)
        elif movement == 2:
            self.client.move(forward=-1.0)
        elif movement == 3:
            self.client.move(strafe=-1.0)
        elif movement == 4:
            self.client.move(strafe=1.0)
        elif movement == 5:
            self.client.move(jump=True)
        elif movement == 6:
            self.client.move(sneak=True)
        elif movement == 7:
            self.client.move(sprint=True)
    
    def _execute_camera(self, camera: int) -> None:
        """Execute camera action."""
        pos = self.client.get_position()
        if pos is None:
            return
        
        # Calculate look direction changes
        if camera == 0:
            pass  # No rotation
        elif camera == 1:  # Look up
            self.client.look_at(pos.x, pos.y + 10, pos.z)
        elif camera == 2:  # Look down
            self.client.look_at(pos.x, pos.y - 10, pos.z)
        elif camera == 3:  # Look left
            self.client.look_at(pos.x - 10, pos.y, pos.z)
        elif camera == 4:  # Look right
            self.client.look_at(pos.x + 10, pos.y, pos.z)
    
    def _execute_interaction(self, interaction: int) -> None:
        """Execute interaction action."""
        if interaction == 0:
            pass  # No interaction
        elif interaction == 1:
            self.client.attack()
        elif interaction == 2:
            self.client.use_item()
        # Add more interactions as needed
    
    def _calculate_reward(self) -> float:
        """Calculate reward for the current step."""
        reward = 0.0
        
        # Negative reward for damage
        current_health = self.client.get_health()
        if current_health < self._prev_health:
            damage = self._prev_health - current_health
            reward -= damage * 0.5
        
        # Small positive reward for exploration
        current_pos = self.client.get_position()
        if current_pos and self._prev_position:
            distance = current_pos.distance_to(self._prev_position)
            reward += distance * 0.01
        
        # Reward for being in The End
        if self.client.get_dimension() == "minecraft:the_end":
            reward += 0.001
        
        return reward
    
    def _check_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Death
        if self.client.get_health() <= 0:
            return True
        
        # Disconnection
        if not self.client.is_connected():
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info dictionary."""
        pos = self.client.get_position()
        
        return {
            'health': self.client.get_health(),
            'food': self.client.get_food(),
            'position': [pos.x, pos.y, pos.z] if pos else [0, 0, 0],
            'dimension': self.client.get_dimension(),
            'step_count': self._step_count,
            'episode_reward': self._episode_reward,
        }
    
    def close(self) -> None:
        """Close the environment."""
        if self.client.is_connected():
            self.client.disconnect()
    
    def render(self, mode: str = 'human') -> None:
        """Render (no-op for live environment)."""
        pass
