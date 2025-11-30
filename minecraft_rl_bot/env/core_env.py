"""
core_env.py - Mock Minecraft environment for reinforcement learning.

This module provides a simplified Minecraft-like environment that can be used
for training RL agents. It simulates:
- 3D world with blocks
- Agent movement and interactions
- Inventory management
- Day/night cycles
- Basic mob spawning

The environment follows the OpenAI Gym interface pattern for compatibility
with standard RL training loops.

TO PLUG IN REAL MINECRAFT:
Replace the MinecraftEnv class methods with calls to:
- Project Malmo (Microsoft's Minecraft AI platform)
- MineRL (Minecraft RL benchmark)
- Custom mod API

The observation and action interfaces are designed to be compatible with
these real Minecraft interfaces.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, field
import random
from enum import IntEnum

from .blocks import BlockType, BLOCK_PROPERTIES, is_solid, get_block_hardness
from .actions import (
    DiscreteAction, ContinuousAction, MovementAction, CameraAction,
    InteractionAction, InventoryAction, DISCRETE_CAMERA_ANGLE,
    WALK_SPEED, SPRINT_SPEED, SNEAK_SPEED
)


class Dimension(IntEnum):
    """Game dimensions."""
    OVERWORLD = 0
    NETHER = 1
    END = 2


@dataclass
class AgentState:
    """
    Complete state of the agent in the environment.
    
    Attributes:
        position: (x, y, z) coordinates in the world
        rotation: (pitch, yaw) camera angles in degrees
        health: Current health points (0-20)
        hunger: Current hunger points (0-20)
        inventory: Dict mapping item name to count
        selected_slot: Currently selected hotbar slot (0-8)
        dimension: Current dimension (Overworld, Nether, End)
        on_ground: Whether agent is standing on solid ground
        in_water: Whether agent is in water
        in_lava: Whether agent is in lava
    """
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 64.0, 0.0]))
    rotation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    health: float = 20.0
    hunger: float = 20.0
    inventory: Dict[str, int] = field(default_factory=dict)
    hotbar: List[Optional[str]] = field(default_factory=lambda: [None] * 9)
    selected_slot: int = 0
    dimension: Dimension = Dimension.OVERWORLD
    on_ground: bool = True
    in_water: bool = False
    in_lava: bool = False
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))


@dataclass
class Entity:
    """Represents an entity in the world (mob, item, etc.)."""
    entity_type: str
    position: np.ndarray
    health: float = 10.0
    hostile: bool = False
    
    
class MinecraftEnv:
    """
    Mock Minecraft environment for RL training.
    
    This environment simulates a simplified Minecraft world with:
    - Procedurally generated terrain
    - Block mining and placement
    - Basic crafting system
    - Survival mechanics (health, hunger)
    - Multiple dimensions (Overworld, Nether, End)
    
    Observation Space:
    - blocks: 3D tensor of shape (21, 21, 21) representing blocks around agent
    - inventory: Dict of item counts
    - agent_state: Health, hunger, position, etc.
    
    Action Space:
    - Discrete: Multi-discrete (movement, camera, interaction, inventory)
    - Continuous: 7-dimensional vector for fine control
    
    To use with real Minecraft:
    1. Inherit from this class
    2. Override reset(), step(), _get_observation() methods
    3. Connect to Minecraft via Malmo/MineRL/mod API
    """
    
    # Observation region size (centered on agent)
    OBS_SIZE = 21
    
    # World boundaries
    WORLD_SIZE = 256
    WORLD_HEIGHT = 128
    
    def __init__(
        self,
        seed: Optional[int] = None,
        continuous_actions: bool = False,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the Minecraft environment.
        
        Args:
            seed: Random seed for world generation (None for random)
            continuous_actions: Whether to use continuous action space
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        self.continuous_actions = continuous_actions
        self.render_mode = render_mode
        
        # Set random seed
        self.seed = seed if seed is not None else random.randint(0, 2**31)
        self.rng = np.random.RandomState(self.seed)
        
        # World state
        self.world: Dict[Tuple[int, int, int], BlockType] = {}
        self.entities: List[Entity] = []
        
        # Agent state
        self.agent = AgentState()
        
        # Game progress tracking
        self.game_tick = 0
        self.day_count = 0
        self.in_nether = False
        self.in_end = False
        self.dragon_defeated = False
        self.game_complete = False
        
        # Statistics for reward calculation
        self.stats = {
            'blocks_mined': 0,
            'items_crafted': 0,
            'mobs_killed': 0,
            'damage_taken': 0,
            'distance_traveled': 0.0
        }
        
        # Initialize world
        self._generate_world()
    
    def _generate_world(self) -> None:
        """
        Generate a procedural Minecraft-like world.
        
        This creates a simplified world with:
        - Terrain with varying heights
        - Underground caves and ores
        - Trees and vegetation
        - Nether portal location
        - End portal location
        
        In a real Minecraft integration, this would be replaced by
        the actual Minecraft world generation.
        """
        # Clear existing world
        self.world.clear()
        
        # Generate terrain heightmap using simple noise
        heightmap = self._generate_heightmap()
        
        # Fill world with blocks
        for x in range(self.WORLD_SIZE):
            for z in range(self.WORLD_SIZE):
                height = heightmap[x, z]
                
                # Bedrock layer
                self.world[(x, 0, z)] = BlockType.BEDROCK
                
                # Stone layers
                for y in range(1, height - 3):
                    # Add occasional ores
                    if self.rng.random() < 0.02:
                        ore = self.rng.choice([
                            BlockType.COAL_ORE,
                            BlockType.IRON_ORE,
                            BlockType.GOLD_ORE,
                            BlockType.DIAMOND_ORE
                        ], p=[0.5, 0.35, 0.1, 0.05])
                        self.world[(x, y, z)] = ore
                    else:
                        self.world[(x, y, z)] = BlockType.STONE
                
                # Dirt layers
                for y in range(max(1, height - 3), height):
                    self.world[(x, y, z)] = BlockType.DIRT
                
                # Surface layer (grass)
                if height > 0:
                    self.world[(x, height, z)] = BlockType.GRASS
                
                # Occasionally add trees
                if self.rng.random() < 0.01 and height > 50:
                    self._place_tree(x, height + 1, z)
        
        # Add water at sea level
        sea_level = 62
        for x in range(self.WORLD_SIZE):
            for z in range(self.WORLD_SIZE):
                if heightmap[x, z] < sea_level:
                    for y in range(heightmap[x, z] + 1, sea_level + 1):
                        self.world[(x, y, z)] = BlockType.WATER
        
        # Place end portal (simplified - just mark location)
        stronghold_x = self.WORLD_SIZE // 2 + self.rng.randint(-50, 50)
        stronghold_z = self.WORLD_SIZE // 2 + self.rng.randint(-50, 50)
        self.end_portal_location = (stronghold_x, 30, stronghold_z)
        
        # Place nether portal frame location
        self.nether_portal_location = (
            self.WORLD_SIZE // 2,
            heightmap[self.WORLD_SIZE // 2, self.WORLD_SIZE // 2] + 1,
            self.WORLD_SIZE // 2
        )
    
    def _generate_heightmap(self) -> np.ndarray:
        """Generate terrain heightmap using simple noise."""
        heightmap = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE), dtype=np.int32)
        
        # Simple multi-octave noise for terrain
        for x in range(self.WORLD_SIZE):
            for z in range(self.WORLD_SIZE):
                # Base height
                height = 64
                
                # Add noise at different scales
                height += int(10 * np.sin(x * 0.02) * np.cos(z * 0.02))
                height += int(5 * np.sin(x * 0.05) * np.cos(z * 0.07))
                height += self.rng.randint(-2, 3)
                
                heightmap[x, z] = max(1, min(height, self.WORLD_HEIGHT - 10))
        
        return heightmap
    
    def _place_tree(self, x: int, y: int, z: int) -> None:
        """Place a simple tree at the given location."""
        # Trunk
        for dy in range(5):
            self.world[(x, y + dy, z)] = BlockType.WOOD_LOG
        
        # Leaves (simple sphere-ish shape)
        for dx in range(-2, 3):
            for dy in range(3, 6):
                for dz in range(-2, 3):
                    if abs(dx) + abs(dz) <= 3:
                        pos = (x + dx, y + dy, z + dz)
                        if pos not in self.world or self.world[pos] == BlockType.AIR:
                            self.world[pos] = BlockType.LEAVES
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: New random seed (optional)
            options: Additional reset options
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        # Update seed if provided
        if seed is not None:
            self.seed = seed
            self.rng = np.random.RandomState(self.seed)
        
        # Regenerate world
        self._generate_world()
        
        # Reset agent state
        spawn_x = self.WORLD_SIZE // 2
        spawn_z = self.WORLD_SIZE // 2
        spawn_y = self._get_surface_height(spawn_x, spawn_z) + 1
        
        self.agent = AgentState(
            position=np.array([float(spawn_x), float(spawn_y), float(spawn_z)]),
            rotation=np.array([0.0, 0.0])
        )
        
        # Reset game state
        self.game_tick = 0
        self.day_count = 0
        self.in_nether = False
        self.in_end = False
        self.dragon_defeated = False
        self.game_complete = False
        self.entities.clear()
        
        # Reset statistics
        self.stats = {
            'blocks_mined': 0,
            'items_crafted': 0,
            'mobs_killed': 0,
            'damage_taken': 0,
            'distance_traveled': 0.0
        }
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self,
        action: DiscreteAction | ContinuousAction | Dict
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Agent's action (DiscreteAction, ContinuousAction, or dict)
            
        Returns:
            observation: New observation after action
            reward: Reward for this step
            terminated: Whether episode ended (death or win)
            truncated: Whether episode was cut short
            info: Additional information
        """
        # Convert action to internal format
        if isinstance(action, dict):
            action = DiscreteAction.from_dict(action)
        
        # Store previous state for reward calculation
        prev_health = self.agent.health
        prev_position = self.agent.position.copy()
        
        # Execute action
        self._execute_action(action)
        
        # Update world state (time, mobs, etc.)
        self._update_world()
        
        # Calculate reward
        reward = self._calculate_step_reward(prev_health, prev_position)
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        if self.agent.health <= 0:
            terminated = True
            reward -= 100  # Death penalty
        elif self.game_complete:
            terminated = True
            reward += 1000  # Win bonus
        
        # Increment tick
        self.game_tick += 1
        
        # Update day count
        if self.game_tick % 24000 == 0:  # Minecraft day length in ticks
            self.day_count += 1
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _execute_action(self, action: DiscreteAction | ContinuousAction) -> None:
        """Execute the given action in the environment."""
        if isinstance(action, ContinuousAction):
            self._execute_continuous_action(action)
        else:
            self._execute_discrete_action(action)
    
    def _execute_discrete_action(self, action: DiscreteAction) -> None:
        """Execute a discrete action."""
        # Get action values, handling both enum and int types
        movement_val = action.movement if isinstance(action.movement, int) else action.movement.value
        camera_val = action.camera if isinstance(action.camera, int) else action.camera.value
        interaction_val = action.interaction if isinstance(action.interaction, int) else action.interaction.value
        inventory_val = action.inventory if isinstance(action.inventory, int) else action.inventory.value
        
        # Handle movement
        if movement_val != 0:  # Not NONE
            self._handle_movement(MovementAction(movement_val))
        
        # Handle camera rotation
        if camera_val != 0:  # Not NONE
            self._handle_camera(CameraAction(camera_val))
        
        # Handle interactions
        if interaction_val != 0:  # Not NONE
            self._handle_interaction(InteractionAction(interaction_val))
        
        # Handle inventory
        if inventory_val != 0:  # Not NONE
            self._handle_inventory(InventoryAction(inventory_val))
    
    def _execute_continuous_action(self, action: ContinuousAction) -> None:
        """Execute a continuous action."""
        # Apply movement
        if abs(action.move_forward) > 0.1 or abs(action.move_strafe) > 0.1:
            # Calculate movement vector based on camera direction
            yaw_rad = np.radians(self.agent.rotation[1])
            
            forward = np.array([
                -np.sin(yaw_rad),
                0,
                np.cos(yaw_rad)
            ])
            strafe = np.array([
                np.cos(yaw_rad),
                0,
                np.sin(yaw_rad)
            ])
            
            move_vector = forward * action.move_forward + strafe * action.move_strafe
            move_vector *= WALK_SPEED * 0.05  # Scale for tick rate
            
            new_pos = self.agent.position + move_vector
            if self._is_valid_position(new_pos):
                self.agent.position = new_pos
        
        # Apply camera rotation
        self.agent.rotation[0] = np.clip(
            self.agent.rotation[0] + action.camera_pitch * 0.1,
            -90, 90
        )
        self.agent.rotation[1] = (self.agent.rotation[1] + action.camera_yaw * 0.1) % 360
        
        # Handle jump
        if action.jump > 0.5 and self.agent.on_ground:
            self.agent.velocity[1] = 8.0  # Jump velocity
        
        # Handle interaction
        if action.interaction > 0:
            self._handle_interaction(InteractionAction(int(action.interaction)))
        
        # Handle inventory slot selection
        self.agent.selected_slot = int(action.inventory_slot)
    
    def _handle_movement(self, movement: MovementAction) -> None:
        """Handle discrete movement action."""
        speed = WALK_SPEED * 0.05  # Scale for tick rate
        
        if movement == MovementAction.SPRINT:
            speed = SPRINT_SPEED * 0.05
        elif movement == MovementAction.SNEAK:
            speed = SNEAK_SPEED * 0.05
        
        yaw_rad = np.radians(self.agent.rotation[1])
        
        move_vector = np.array([0.0, 0.0, 0.0])
        
        if movement in (MovementAction.FORWARD, MovementAction.SPRINT):
            move_vector = np.array([
                -np.sin(yaw_rad) * speed,
                0,
                np.cos(yaw_rad) * speed
            ])
        elif movement == MovementAction.BACKWARD:
            move_vector = np.array([
                np.sin(yaw_rad) * speed,
                0,
                -np.cos(yaw_rad) * speed
            ])
        elif movement == MovementAction.LEFT:
            move_vector = np.array([
                np.cos(yaw_rad) * speed,
                0,
                np.sin(yaw_rad) * speed
            ])
        elif movement == MovementAction.RIGHT:
            move_vector = np.array([
                -np.cos(yaw_rad) * speed,
                0,
                -np.sin(yaw_rad) * speed
            ])
        elif movement == MovementAction.JUMP and self.agent.on_ground:
            self.agent.velocity[1] = 8.0
        
        new_pos = self.agent.position + move_vector
        if self._is_valid_position(new_pos):
            distance = np.linalg.norm(move_vector)
            self.stats['distance_traveled'] += distance
            self.agent.position = new_pos
    
    def _handle_camera(self, camera: CameraAction) -> None:
        """Handle discrete camera action."""
        angle = DISCRETE_CAMERA_ANGLE
        
        if camera == CameraAction.LOOK_UP:
            self.agent.rotation[0] = max(-90, self.agent.rotation[0] - angle)
        elif camera == CameraAction.LOOK_DOWN:
            self.agent.rotation[0] = min(90, self.agent.rotation[0] + angle)
        elif camera == CameraAction.LOOK_LEFT:
            self.agent.rotation[1] = (self.agent.rotation[1] - angle) % 360
        elif camera == CameraAction.LOOK_RIGHT:
            self.agent.rotation[1] = (self.agent.rotation[1] + angle) % 360
    
    def _handle_interaction(self, interaction: InteractionAction) -> None:
        """Handle interaction action."""
        if interaction == InteractionAction.MINE:
            self._mine_block()
        elif interaction == InteractionAction.PLACE:
            self._place_block()
        elif interaction == InteractionAction.USE:
            self._use_item()
        elif interaction == InteractionAction.ATTACK:
            self._attack_entity()
        elif interaction == InteractionAction.CRAFT:
            self._open_crafting()
    
    def _handle_inventory(self, inventory: InventoryAction) -> None:
        """Handle inventory action."""
        if InventoryAction.SELECT_SLOT_1 <= inventory <= InventoryAction.SELECT_SLOT_9:
            self.agent.selected_slot = inventory.value - 1
        elif inventory == InventoryAction.DROP_ITEM:
            self._drop_item()
    
    def _mine_block(self) -> None:
        """Mine the block the agent is looking at."""
        target = self._get_target_block()
        if target is None:
            return
        
        block_type = self.world.get(target, BlockType.AIR)
        if block_type == BlockType.AIR:
            return
        
        props = BLOCK_PROPERTIES.get(block_type)
        if props is None or props.hardness < 0:
            return  # Unbreakable
        
        # Get drops and add to inventory
        for item_name, count in props.drops:
            self.agent.inventory[item_name] = self.agent.inventory.get(item_name, 0) + count
        
        # Remove block
        self.world[target] = BlockType.AIR
        self.stats['blocks_mined'] += 1
    
    def _place_block(self) -> None:
        """Place a block from inventory."""
        # Simplified: just check if we have cobblestone
        if self.agent.inventory.get('cobblestone', 0) > 0:
            target = self._get_adjacent_block()
            if target and self.world.get(target, BlockType.AIR) == BlockType.AIR:
                self.world[target] = BlockType.COBBLESTONE
                self.agent.inventory['cobblestone'] -= 1
    
    def _use_item(self) -> None:
        """Use the currently selected item."""
        # Check for special interactions
        target = self._get_target_block()
        if target:
            block_type = self.world.get(target, BlockType.AIR)
            
            # Enter nether portal
            if block_type == BlockType.NETHER_PORTAL and not self.in_nether:
                self.in_nether = True
                self.agent.dimension = Dimension.NETHER
            
            # Enter end portal
            elif block_type == BlockType.END_PORTAL and not self.in_end:
                self.in_end = True
                self.agent.dimension = Dimension.END
    
    def _attack_entity(self) -> None:
        """Attack an entity in front of the agent."""
        # Find nearest entity in attack range
        attack_range = 3.0
        for entity in self.entities[:]:
            dist = np.linalg.norm(entity.position - self.agent.position)
            if dist < attack_range:
                entity.health -= 4  # Base damage
                if entity.health <= 0:
                    self.entities.remove(entity)
                    self.stats['mobs_killed'] += 1
                    
                    # Special case: Ender Dragon
                    if entity.entity_type == 'ender_dragon':
                        self.dragon_defeated = True
                        self.game_complete = True
                break
    
    def _open_crafting(self) -> None:
        """Simplified crafting system."""
        inv = self.agent.inventory
        
        # Wood log -> planks
        if inv.get('wood_log', 0) >= 1:
            inv['wood_log'] -= 1
            inv['wood_planks'] = inv.get('wood_planks', 0) + 4
            self.stats['items_crafted'] += 1
        
        # Planks -> sticks
        elif inv.get('wood_planks', 0) >= 2:
            inv['wood_planks'] -= 2
            inv['stick'] = inv.get('stick', 0) + 4
            self.stats['items_crafted'] += 1
        
        # Planks + sticks -> wooden pickaxe
        elif inv.get('wood_planks', 0) >= 3 and inv.get('stick', 0) >= 2:
            inv['wood_planks'] -= 3
            inv['stick'] -= 2
            inv['wooden_pickaxe'] = inv.get('wooden_pickaxe', 0) + 1
            self.stats['items_crafted'] += 1
        
        # Cobblestone + sticks -> stone pickaxe
        elif inv.get('cobblestone', 0) >= 3 and inv.get('stick', 0) >= 2:
            inv['cobblestone'] -= 3
            inv['stick'] -= 2
            inv['stone_pickaxe'] = inv.get('stone_pickaxe', 0) + 1
            self.stats['items_crafted'] += 1
    
    def _drop_item(self) -> None:
        """Drop the currently held item."""
        slot_item = self.agent.hotbar[self.agent.selected_slot]
        if slot_item and self.agent.inventory.get(slot_item, 0) > 0:
            self.agent.inventory[slot_item] -= 1
    
    def _get_target_block(self) -> Optional[Tuple[int, int, int]]:
        """Get the block position the agent is looking at."""
        # Ray cast from agent position in look direction
        pitch_rad = np.radians(self.agent.rotation[0])
        yaw_rad = np.radians(self.agent.rotation[1])
        
        direction = np.array([
            -np.sin(yaw_rad) * np.cos(pitch_rad),
            -np.sin(pitch_rad),
            np.cos(yaw_rad) * np.cos(pitch_rad)
        ])
        
        # Check blocks along ray
        for dist in np.arange(0.5, 5.0, 0.5):
            pos = self.agent.position + direction * dist
            block_pos = tuple(np.floor(pos).astype(int))
            if self.world.get(block_pos, BlockType.AIR) != BlockType.AIR:
                return block_pos
        
        return None
    
    def _get_adjacent_block(self) -> Optional[Tuple[int, int, int]]:
        """Get an empty block position adjacent to target."""
        target = self._get_target_block()
        if target is None:
            # Place in front of agent
            yaw_rad = np.radians(self.agent.rotation[1])
            front = self.agent.position + np.array([
                -np.sin(yaw_rad) * 2,
                0,
                np.cos(yaw_rad) * 2
            ])
            return tuple(np.floor(front).astype(int))
        
        # Find adjacent empty space
        for offset in [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)]:
            adjacent = (target[0] + offset[0], target[1] + offset[1], target[2] + offset[2])
            if self.world.get(adjacent, BlockType.AIR) == BlockType.AIR:
                return adjacent
        
        return None
    
    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """Check if a position is valid for the agent to move to."""
        # Check world boundaries
        if not (0 <= pos[0] < self.WORLD_SIZE and 
                0 <= pos[1] < self.WORLD_HEIGHT and 
                0 <= pos[2] < self.WORLD_SIZE):
            return False
        
        # Check for solid blocks at feet and head level
        feet_pos = tuple(np.floor(pos).astype(int))
        head_pos = (feet_pos[0], feet_pos[1] + 1, feet_pos[2])
        
        feet_block = self.world.get(feet_pos, BlockType.AIR)
        head_block = self.world.get(head_pos, BlockType.AIR)
        
        return not is_solid(feet_block) and not is_solid(head_block)
    
    def _get_surface_height(self, x: int, z: int) -> int:
        """Get the surface height at given x, z coordinates."""
        for y in range(self.WORLD_HEIGHT - 1, 0, -1):
            if is_solid(self.world.get((x, y, z), BlockType.AIR)):
                return y
        return 0
    
    def _update_world(self) -> None:
        """Update world state (physics, mobs, etc.)."""
        # Apply gravity
        if not self.agent.on_ground:
            self.agent.velocity[1] -= 0.08  # Gravity
            self.agent.velocity[1] = max(self.agent.velocity[1], -3.92)  # Terminal velocity
        
        # Apply vertical velocity
        new_y = self.agent.position[1] + self.agent.velocity[1] * 0.05
        feet_pos = (
            int(np.floor(self.agent.position[0])),
            int(np.floor(new_y)),
            int(np.floor(self.agent.position[2]))
        )
        
        if is_solid(self.world.get(feet_pos, BlockType.AIR)):
            self.agent.on_ground = True
            self.agent.velocity[1] = 0
        else:
            self.agent.position[1] = new_y
            self.agent.on_ground = False
        
        # Check for hazards
        self._check_hazards()
        
        # Spawn/update mobs occasionally
        if self.game_tick % 100 == 0:
            self._update_mobs()
    
    def _check_hazards(self) -> None:
        """Check for environmental hazards (lava, fall damage, etc.)."""
        pos = tuple(np.floor(self.agent.position).astype(int))
        block = self.world.get(pos, BlockType.AIR)
        
        # Lava damage
        if block == BlockType.LAVA:
            self.agent.in_lava = True
            self.agent.health -= 4
            self.stats['damage_taken'] += 4
        else:
            self.agent.in_lava = False
        
        # Water (no damage, but track state)
        self.agent.in_water = block == BlockType.WATER
    
    def _update_mobs(self) -> None:
        """Spawn and update mob entities."""
        # Spawn Ender Dragon in End dimension
        if self.in_end and not self.dragon_defeated:
            has_dragon = any(e.entity_type == 'ender_dragon' for e in self.entities)
            if not has_dragon:
                dragon_pos = self.agent.position + np.array([0, 50, 0])
                self.entities.append(Entity(
                    entity_type='ender_dragon',
                    position=dragon_pos,
                    health=200,
                    hostile=True
                ))
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get the current observation.
        
        Returns a dictionary containing:
        - blocks: 3D tensor of block types (21x21x21)
        - inventory: Array of item counts
        - agent_state: Array of agent state values
        
        This format is designed to be compatible with both the mock
        environment and real Minecraft integrations.
        """
        # Get 3D block observation centered on agent
        blocks = np.zeros((self.OBS_SIZE, self.OBS_SIZE, self.OBS_SIZE), dtype=np.int32)
        
        center = np.floor(self.agent.position).astype(int)
        half_size = self.OBS_SIZE // 2
        
        for dx in range(-half_size, half_size + 1):
            for dy in range(-half_size, half_size + 1):
                for dz in range(-half_size, half_size + 1):
                    world_pos = (center[0] + dx, center[1] + dy, center[2] + dz)
                    block_type = self.world.get(world_pos, BlockType.AIR)
                    
                    obs_x = dx + half_size
                    obs_y = dy + half_size
                    obs_z = dz + half_size
                    
                    # Handle both enum and integer block types
                    if hasattr(block_type, 'value'):
                        blocks[obs_x, obs_y, obs_z] = block_type.value
                    else:
                        blocks[obs_x, obs_y, obs_z] = int(block_type)
        
        # Inventory observation (simplified)
        inventory_items = [
            'wood_log', 'wood_planks', 'cobblestone', 'coal', 'iron_ore',
            'diamond', 'stick', 'wooden_pickaxe', 'stone_pickaxe', 'iron_pickaxe'
        ]
        inventory = np.array([
            self.agent.inventory.get(item, 0) for item in inventory_items
        ], dtype=np.float32)
        
        # Agent state observation
        agent_state = np.array([
            self.agent.health / 20.0,
            self.agent.hunger / 20.0,
            self.agent.position[0] / self.WORLD_SIZE,
            self.agent.position[1] / self.WORLD_HEIGHT,
            self.agent.position[2] / self.WORLD_SIZE,
            self.agent.rotation[0] / 90.0,
            self.agent.rotation[1] / 360.0,
            float(self.agent.on_ground),
            float(self.agent.in_water),
            float(self.agent.in_lava),
            float(self.in_nether),
            float(self.in_end),
            float(self.dragon_defeated)
        ], dtype=np.float32)
        
        return {
            'blocks': blocks,
            'inventory': inventory,
            'agent_state': agent_state
        }
    
    def _calculate_step_reward(
        self,
        prev_health: float,
        prev_position: np.ndarray
    ) -> float:
        """
        Calculate reward for the current step.
        
        Reward signals:
        - Negative: Taking damage
        - Positive: Mining valuable blocks, crafting, progression
        
        The reward structure is designed to encourage:
        1. Survival (avoiding damage)
        2. Resource gathering
        3. Crafting progression
        4. Game completion
        """
        reward = 0.0
        
        # Damage penalty
        damage = prev_health - self.agent.health
        if damage > 0:
            reward -= damage * 2.0
        
        # Small reward for exploration (distance traveled)
        distance = np.linalg.norm(self.agent.position - prev_position)
        reward += distance * 0.01
        
        return reward
    
    def _get_info(self) -> Dict:
        """Get additional information about the environment state."""
        return {
            'game_tick': self.game_tick,
            'day_count': self.day_count,
            'dimension': self.agent.dimension.name,
            'health': self.agent.health,
            'hunger': self.agent.hunger,
            'position': self.agent.position.tolist(),
            'stats': self.stats.copy(),
            'in_nether': self.in_nether,
            'in_end': self.in_end,
            'dragon_defeated': self.dragon_defeated,
            'game_complete': self.game_complete
        }
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        In a real Minecraft integration, this would capture the game screen.
        For the mock environment, this returns a simple text representation.
        """
        if self.render_mode == 'human':
            print(f"Position: {self.agent.position}")
            print(f"Health: {self.agent.health}, Hunger: {self.agent.hunger}")
            print(f"Inventory: {self.agent.inventory}")
            print(f"Dimension: {self.agent.dimension.name}")
        elif self.render_mode == 'rgb_array':
            # Return a simple visualization array
            return np.zeros((64, 64, 3), dtype=np.uint8)
        
        return None
    
    def close(self) -> None:
        """Clean up environment resources."""
        pass
