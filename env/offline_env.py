"""
offline_env.py - Mock Minecraft environment for training and testing.

This module provides a simplified Minecraft-like environment that:
- Simulates core game mechanics (blocks, inventory, health)
- Provides observations compatible with the RL agent
- Supports both discrete and continuous action spaces

This environment is used for:
1. Training the RL agent without needing a real Minecraft instance
2. Testing and debugging agent behavior
3. Validating the training pipeline
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass, field
from enum import IntEnum


class BlockType(IntEnum):
    """Block types in the environment."""
    AIR = 0
    STONE = 1
    DIRT = 2
    GRASS = 3
    WOOD_LOG = 4
    LEAVES = 5
    COBBLESTONE = 6
    COAL_ORE = 7
    IRON_ORE = 8
    GOLD_ORE = 9
    DIAMOND_ORE = 10
    OBSIDIAN = 11
    WATER = 12
    LAVA = 13
    SAND = 14
    GRAVEL = 15
    CRAFTING_TABLE = 16
    FURNACE = 17
    CHEST = 18
    BEDROCK = 19
    NETHER_PORTAL = 20
    END_PORTAL = 21
    END_STONE = 22
    PURPUR = 23
    END_STONE_BRICKS = 24
    END_ROD = 25


@dataclass
class DiscreteAction:
    """
    Discrete action representation for the environment.
    
    Each component is an integer index into its action space:
    - movement: 0=none, 1=forward, 2=back, 3=left, 4=right, 5=jump, 6=sneak, 7=sprint
    - camera: 0=none, 1=up, 2=down, 3=left, 4=right
    - interaction: 0=none, 1=attack, 2=use, 3=mine, 4=place, 5=craft, 6=open
    - inventory: 0=none, 1-11=hotbar slots
    """
    movement: int = 0
    camera: int = 0
    interaction: int = 0
    inventory: int = 0


@dataclass
class ObservationSpace:
    """Observation space definition."""
    blocks_shape: Tuple[int, int, int] = (21, 21, 21)
    inventory_size: int = 10
    agent_state_size: int = 13


@dataclass
class ActionSpace:
    """Action space definition."""
    movement_dim: int = 8
    camera_dim: int = 5
    interaction_dim: int = 7
    inventory_dim: int = 12


class MinecraftEnv:
    """
    Mock Minecraft environment for RL training.
    
    This environment simulates a simplified version of Minecraft
    suitable for training RL agents. It provides:
    - 3D block observations around the agent
    - Inventory management
    - Health and hunger mechanics
    - Day/night cycles
    - Basic crafting and mining
    
    Usage:
        env = MinecraftEnv(seed=42)
        obs, info = env.reset()
        action = DiscreteAction(movement=1, interaction=3)
        obs, reward, terminated, truncated, info = env.step(action)
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        continuous_actions: bool = False,
        obs_size: int = 21,
        max_steps: int = 10000
    ):
        """
        Initialize the environment.
        
        Args:
            seed: Random seed for reproducibility
            continuous_actions: Use continuous action space
            obs_size: Size of observation region (obs_size x obs_size x obs_size)
            max_steps: Maximum steps before truncation
        """
        self.seed_value = seed
        self.continuous_actions = continuous_actions
        self.obs_size = obs_size
        self.max_steps = max_steps
        
        # Observation and action spaces
        self.observation_space = ObservationSpace(
            blocks_shape=(obs_size, obs_size, obs_size),
            inventory_size=10,
            agent_state_size=13
        )
        self.action_space = ActionSpace()
        
        # World state
        self._world: np.ndarray = None
        self._position: np.ndarray = None
        self._velocity: np.ndarray = None
        self._yaw: float = 0.0
        self._pitch: float = 0.0
        
        # Agent state
        self._health: float = 20.0
        self._hunger: float = 20.0
        self._inventory: np.ndarray = None
        
        # Game state
        self._time: int = 0
        self._day_count: int = 0
        self._step_count: int = 0
        self._in_nether: bool = False
        self._in_end: bool = False
        self._dragon_defeated: bool = False
        self._game_complete: bool = False
        
        # Statistics
        self._stats: Dict[str, int] = {}
        
        # Random number generator
        self._rng: np.random.Generator = None
        
        # Initialize
        self.reset(seed=seed)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed (optional)
            options: Additional options (optional)
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        if seed is not None:
            self.seed_value = seed
        
        if self.seed_value is not None:
            self._rng = np.random.default_rng(self.seed_value)
        else:
            self._rng = np.random.default_rng()
        
        # Generate world
        self._generate_world()
        
        # Reset agent position (spawn at center, on surface)
        world_center = np.array([64.0, 64.0, 64.0])
        self._position = world_center.copy()
        self._velocity = np.zeros(3)
        self._yaw = 0.0
        self._pitch = 0.0
        
        # Reset agent state
        self._health = 20.0
        self._hunger = 20.0
        self._inventory = np.zeros(self.observation_space.inventory_size, dtype=np.float32)
        
        # Reset game state
        self._time = 0
        self._day_count = 0
        self._step_count = 0
        self._in_nether = False
        self._in_end = False
        self._dragon_defeated = False
        self._game_complete = False
        
        # Reset statistics
        self._stats = {
            'blocks_mined': 0,
            'items_crafted': 0,
            'mobs_killed': 0,
            'damage_taken': 0
        }
        
        return self._get_observation(), self._get_info()
    
    def step(
        self,
        action: DiscreteAction
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Action to execute
            
        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode ended (success/failure)
            truncated: Whether episode was truncated (time limit)
            info: Additional information
        """
        self._step_count += 1
        prev_info = self._get_info()
        
        # Process action
        reward = self._process_action(action)
        
        # Update world state
        self._update_world()
        
        # Update agent state
        self._update_agent()
        
        # Check termination conditions
        terminated = self._check_terminated()
        truncated = self._step_count >= self.max_steps
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _generate_world(self) -> None:
        """Generate a procedural world."""
        world_size = 128
        self._world = np.zeros((world_size, world_size, world_size), dtype=np.int32)
        
        # Generate terrain (simplified)
        for x in range(world_size):
            for z in range(world_size):
                # Height varies based on noise-like function
                height = 60 + int(5 * np.sin(x * 0.1) * np.cos(z * 0.1))
                
                # Fill layers
                for y in range(height):
                    if y < 5:
                        self._world[x, y, z] = BlockType.BEDROCK
                    elif y < height - 3:
                        self._world[x, y, z] = BlockType.STONE
                    elif y < height - 1:
                        self._world[x, y, z] = BlockType.DIRT
                    else:
                        self._world[x, y, z] = BlockType.GRASS
        
        # Add some ores
        self._add_ore(BlockType.COAL_ORE, count=500, min_y=5, max_y=50)
        self._add_ore(BlockType.IRON_ORE, count=200, min_y=5, max_y=40)
        self._add_ore(BlockType.GOLD_ORE, count=50, min_y=5, max_y=30)
        self._add_ore(BlockType.DIAMOND_ORE, count=20, min_y=5, max_y=15)
        
        # Add trees
        self._add_trees(count=50)
    
    def _add_ore(self, block_type: int, count: int, min_y: int, max_y: int) -> None:
        """Add ore clusters to the world."""
        world_size = self._world.shape[0]
        
        for _ in range(count):
            x = self._rng.integers(5, world_size - 5)
            y = self._rng.integers(min_y, max_y)
            z = self._rng.integers(5, world_size - 5)
            
            if self._world[x, y, z] == BlockType.STONE:
                self._world[x, y, z] = block_type
    
    def _add_trees(self, count: int) -> None:
        """Add trees to the world."""
        world_size = self._world.shape[0]
        
        for _ in range(count):
            x = self._rng.integers(10, world_size - 10)
            z = self._rng.integers(10, world_size - 10)
            
            # Find surface
            for y in range(70, 50, -1):
                if self._world[x, y, z] == BlockType.GRASS:
                    # Add trunk
                    for dy in range(1, 5):
                        if y + dy < world_size:
                            self._world[x, y + dy, z] = BlockType.WOOD_LOG
                    
                    # Add leaves
                    for dx in range(-2, 3):
                        for dz in range(-2, 3):
                            for dy in range(4, 7):
                                if (y + dy < world_size and 
                                    0 <= x + dx < world_size and 
                                    0 <= z + dz < world_size):
                                    if self._world[x + dx, y + dy, z + dz] == BlockType.AIR:
                                        self._world[x + dx, y + dy, z + dz] = BlockType.LEAVES
                    break
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get the current observation."""
        # Get blocks around agent
        blocks = self._get_local_blocks()
        
        # Get inventory
        inventory = self._inventory.copy()
        
        # Get agent state
        agent_state = np.array([
            self._position[0],
            self._position[1],
            self._position[2],
            self._velocity[0],
            self._velocity[1],
            self._velocity[2],
            self._yaw,
            self._pitch,
            self._health,
            self._hunger,
            float(self._in_nether),
            float(self._in_end),
            float(self._day_count)
        ], dtype=np.float32)
        
        return {
            'blocks': blocks,
            'inventory': inventory,
            'agent_state': agent_state
        }
    
    def _get_local_blocks(self) -> np.ndarray:
        """Get blocks around the agent."""
        half_size = self.obs_size // 2
        blocks = np.zeros(
            (self.obs_size, self.obs_size, self.obs_size),
            dtype=np.float32
        )
        
        px, py, pz = int(self._position[0]), int(self._position[1]), int(self._position[2])
        
        for dx in range(-half_size, half_size + 1):
            for dy in range(-half_size, half_size + 1):
                for dz in range(-half_size, half_size + 1):
                    wx, wy, wz = px + dx, py + dy, pz + dz
                    
                    if (0 <= wx < self._world.shape[0] and
                        0 <= wy < self._world.shape[1] and
                        0 <= wz < self._world.shape[2]):
                        blocks[dx + half_size, dy + half_size, dz + half_size] = \
                            self._world[wx, wy, wz]
        
        return blocks
    
    def _get_info(self) -> Dict:
        """Get current game information."""
        return {
            'health': self._health,
            'hunger': self._hunger,
            'position': self._position.copy(),
            'day_count': self._day_count,
            'time': self._time,
            'in_nether': self._in_nether,
            'in_end': self._in_end,
            'dragon_defeated': self._dragon_defeated,
            'game_complete': self._game_complete,
            'stats': self._stats.copy()
        }
    
    def _process_action(self, action: DiscreteAction) -> float:
        """Process agent action and return base reward."""
        reward = 0.0
        
        # Process movement
        self._process_movement(action.movement)
        
        # Process camera
        self._process_camera(action.camera)
        
        # Process interaction
        reward += self._process_interaction(action.interaction)
        
        # Process inventory
        self._process_inventory(action.inventory)
        
        return reward
    
    def _process_movement(self, movement: int) -> None:
        """Process movement action."""
        speed = 0.1
        jump_velocity = 0.5
        
        # Movement direction based on yaw
        forward = np.array([
            np.sin(self._yaw),
            0,
            np.cos(self._yaw)
        ])
        right = np.array([
            np.cos(self._yaw),
            0,
            -np.sin(self._yaw)
        ])
        
        if movement == 1:  # Forward
            self._velocity[:] += forward * speed
        elif movement == 2:  # Back
            self._velocity[:] -= forward * speed
        elif movement == 3:  # Left
            self._velocity[:] -= right * speed
        elif movement == 4:  # Right
            self._velocity[:] += right * speed
        elif movement == 5:  # Jump
            if self._is_on_ground():
                self._velocity[1] = jump_velocity
        elif movement == 6:  # Sneak
            self._velocity[:] *= 0.3
        elif movement == 7:  # Sprint
            self._velocity[:] *= 1.5
        
        # Apply velocity
        self._position += self._velocity
        
        # Apply friction and gravity
        self._velocity[0] *= 0.9
        self._velocity[2] *= 0.9
        self._velocity[1] -= 0.08  # Gravity
        
        # Clamp to world bounds
        self._position = np.clip(
            self._position,
            [1, 1, 1],
            [126, 126, 126]
        )
        
        # Ground collision
        if self._is_on_ground():
            self._velocity[1] = max(0, self._velocity[1])
    
    def _process_camera(self, camera: int) -> None:
        """Process camera action."""
        rotation_speed = 0.1
        
        if camera == 1:  # Up
            self._pitch = max(-1.5, self._pitch - rotation_speed)
        elif camera == 2:  # Down
            self._pitch = min(1.5, self._pitch + rotation_speed)
        elif camera == 3:  # Left
            self._yaw -= rotation_speed
        elif camera == 4:  # Right
            self._yaw += rotation_speed
    
    def _process_interaction(self, interaction: int) -> float:
        """Process interaction action and return reward."""
        reward = 0.0
        
        if interaction == 3:  # Mine
            # Try to mine block in front
            look_pos = self._get_look_position()
            x, y, z = int(look_pos[0]), int(look_pos[1]), int(look_pos[2])
            
            if (0 <= x < self._world.shape[0] and
                0 <= y < self._world.shape[1] and
                0 <= z < self._world.shape[2]):
                
                block = self._world[x, y, z]
                if block != BlockType.AIR and block != BlockType.BEDROCK:
                    self._world[x, y, z] = BlockType.AIR
                    self._stats['blocks_mined'] += 1
                    
                    # Add to inventory based on block type
                    if block == BlockType.COAL_ORE:
                        self._inventory[0] += 1
                        reward = 2.0
                    elif block == BlockType.IRON_ORE:
                        self._inventory[1] += 1
                        reward = 5.0
                    elif block == BlockType.DIAMOND_ORE:
                        self._inventory[2] += 1
                        reward = 20.0
                    else:
                        reward = 0.5
        
        return reward
    
    def _process_inventory(self, inventory: int) -> None:
        """Process inventory action."""
        # Simplified inventory management
        pass
    
    def _get_look_position(self) -> np.ndarray:
        """Get position the agent is looking at."""
        look_distance = 3.0
        
        direction = np.array([
            np.sin(self._yaw) * np.cos(self._pitch),
            -np.sin(self._pitch),
            np.cos(self._yaw) * np.cos(self._pitch)
        ])
        
        return self._position + direction * look_distance
    
    def _is_on_ground(self) -> bool:
        """Check if agent is standing on solid ground."""
        x, y, z = int(self._position[0]), int(self._position[1]) - 1, int(self._position[2])
        
        if (0 <= x < self._world.shape[0] and
            0 <= y < self._world.shape[1] and
            0 <= z < self._world.shape[2]):
            return self._world[x, y, z] != BlockType.AIR
        
        return False
    
    def _update_world(self) -> None:
        """Update world state (time, day/night)."""
        self._time += 1
        
        # Day lasts 24000 ticks
        if self._time >= 24000:
            self._time = 0
            self._day_count += 1
    
    def _update_agent(self) -> None:
        """Update agent state (health, hunger)."""
        # Hunger decreases slowly
        self._hunger -= 0.001
        self._hunger = max(0, self._hunger)
        
        # Health regenerates if well fed
        if self._hunger > 18:
            self._health = min(20, self._health + 0.01)
        
        # Starving damages health
        if self._hunger <= 0:
            self._health -= 0.1
    
    def _check_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Death
        if self._health <= 0:
            return True
        
        # Game complete
        if self._game_complete:
            return True
        
        return False
    
    def close(self) -> None:
        """Clean up environment resources."""
        pass
    
    def render(self) -> None:
        """Render the environment (stub for compatibility)."""
        pass
