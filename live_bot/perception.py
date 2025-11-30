"""
perception.py - World observation and analysis for Elytra Finder Bot.

This module provides perception capabilities:
- World snapshot capture
- Block and entity analysis
- Feature extraction for RL policy
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from integration.mc_client import MinecraftClient, Position, Block, Entity

logger = logging.getLogger(__name__)


@dataclass
class WorldSnapshot:
    """A snapshot of the world around the player."""
    center: Position
    blocks: Dict[Tuple[int, int, int], Block]
    entities: List[Entity]
    radius: int
    dimension: str


class PerceptionModule:
    """
    Perception system for the Minecraft bot.
    
    Handles world observation, block analysis, and feature extraction
    for both scripted behavior and RL policy inference.
    """
    
    # End-specific block types for detection
    END_BLOCKS = {
        "minecraft:end_stone",
        "minecraft:end_stone_bricks",
        "minecraft:purpur_block",
        "minecraft:purpur_pillar",
        "minecraft:purpur_stairs",
        "minecraft:purpur_slab",
        "minecraft:end_rod",
    }
    
    SHIP_BLOCKS = {
        "minecraft:purpur_block",
        "minecraft:purpur_stairs",
        "minecraft:purpur_slab",
        "minecraft:end_rod",
        "minecraft:chest",
        "minecraft:dragon_head",
        "minecraft:brewing_stand",
    }
    
    def __init__(self, client: MinecraftClient):
        """
        Initialize the perception module.
        
        Args:
            client: Minecraft client for world queries
        """
        self.client = client
    
    def get_world_snapshot(
        self,
        radius: int = 64
    ) -> WorldSnapshot:
        """
        Get a snapshot of the world around the player.
        
        Args:
            radius: Radius to capture (in blocks)
            
        Returns:
            WorldSnapshot containing nearby blocks and entities
        """
        pos = self.client.get_position()
        if pos is None:
            pos = Position(0, 64, 0)
        
        # Get nearby blocks
        blocks_list = self.client.get_nearby_blocks(radius)
        blocks_dict = {(b.x, b.y, b.z): b for b in blocks_list}
        
        # Get nearby entities
        entities = self.client.get_nearby_entities(radius)
        
        return WorldSnapshot(
            center=pos,
            blocks=blocks_dict,
            entities=entities,
            radius=radius,
            dimension=self.client.get_dimension()
        )
    
    def get_observation_for_policy(
        self,
        obs_size: int = 21
    ) -> Dict[str, np.ndarray]:
        """
        Get observation formatted for the RL policy.
        
        Returns observation in the format expected by the agent:
        - blocks: (obs_size, obs_size, obs_size) array of block IDs
        - inventory: (inventory_size,) array of item counts
        - agent_state: (state_size,) array of agent state values
        
        Args:
            obs_size: Size of the block observation cube
            
        Returns:
            Dictionary containing observation arrays
        """
        pos = self.client.get_position()
        if pos is None:
            pos = Position(0, 64, 0)
        
        # Block observation
        blocks = self._get_block_observation(pos, obs_size)
        
        # Inventory observation (simplified)
        inventory = self._get_inventory_observation()
        
        # Agent state observation
        agent_state = self._get_agent_state_observation(pos)
        
        return {
            'blocks': blocks,
            'inventory': inventory,
            'agent_state': agent_state
        }
    
    def _get_block_observation(
        self,
        center: Position,
        obs_size: int
    ) -> np.ndarray:
        """Get block observation cube around center position."""
        half_size = obs_size // 2
        blocks = np.zeros((obs_size, obs_size, obs_size), dtype=np.float32)
        
        # Block ID mapping for known blocks
        block_id_map = {
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
        }
        
        cx, cy, cz = int(center.x), int(center.y), int(center.z)
        
        for dx in range(-half_size, half_size + 1):
            for dy in range(-half_size, half_size + 1):
                for dz in range(-half_size, half_size + 1):
                    x, y, z = cx + dx, cy + dy, cz + dz
                    block = self.client.get_block_at(x, y, z)
                    
                    if block:
                        block_id = block_id_map.get(block.block_id, 11)
                    else:
                        block_id = 0
                    
                    bx = dx + half_size
                    by = dy + half_size
                    bz = dz + half_size
                    blocks[bx, by, bz] = block_id
        
        return blocks
    
    def _get_inventory_observation(self) -> np.ndarray:
        """Get simplified inventory observation."""
        inventory = np.zeros(10, dtype=np.float32)
        
        items = self.client.get_inventory()
        
        # Count items by category
        for item in items:
            if item is None:
                continue
            
            if "pickaxe" in item.item_id:
                inventory[0] += item.count
            elif "sword" in item.item_id:
                inventory[1] += item.count
            elif "block" in item.item_id:
                inventory[2] += item.count
            elif "food" in item.item_id or "apple" in item.item_id:
                inventory[3] += item.count
            elif "elytra" in item.item_id:
                inventory[4] += item.count
            elif "ender_pearl" in item.item_id:
                inventory[5] += item.count
            # Other categories...
        
        # Normalize
        inventory = np.clip(inventory / 64.0, 0, 1)
        
        return inventory
    
    def _get_agent_state_observation(
        self,
        pos: Position
    ) -> np.ndarray:
        """Get agent state observation."""
        state = np.zeros(13, dtype=np.float32)
        
        # Position (normalized to chunk scale)
        state[0] = pos.x / 100.0
        state[1] = pos.y / 256.0
        state[2] = pos.z / 100.0
        
        # Velocity (zeros for now, could track)
        state[3] = 0.0
        state[4] = 0.0
        state[5] = 0.0
        
        # Rotation (zeros for now)
        state[6] = 0.0
        state[7] = 0.0
        
        # Health and food
        state[8] = self.client.get_health() / 20.0
        state[9] = self.client.get_food() / 20.0
        
        # Dimension flags
        dimension = self.client.get_dimension()
        state[10] = 1.0 if dimension == "minecraft:the_nether" else 0.0
        state[11] = 1.0 if dimension == "minecraft:the_end" else 0.0
        
        # Day count (zeros for now)
        state[12] = 0.0
        
        return state
    
    def find_blocks_by_type(
        self,
        world_snapshot: WorldSnapshot,
        block_types: set
    ) -> List[Tuple[int, int, int]]:
        """
        Find all blocks of given types in a world snapshot.
        
        Args:
            world_snapshot: World data to search
            block_types: Set of block IDs to find
            
        Returns:
            List of (x, y, z) positions
        """
        results = []
        
        for (x, y, z), block in world_snapshot.blocks.items():
            if block.block_id in block_types:
                results.append((x, y, z))
        
        return results
    
    def count_blocks_in_region(
        self,
        world_snapshot: WorldSnapshot,
        center: Tuple[int, int, int],
        radius: int,
        block_types: set
    ) -> int:
        """
        Count blocks of given types in a region.
        
        Args:
            world_snapshot: World data
            center: Center of region
            radius: Search radius
            block_types: Block types to count
            
        Returns:
            Count of matching blocks
        """
        count = 0
        cx, cy, cz = center
        
        for (x, y, z), block in world_snapshot.blocks.items():
            if (abs(x - cx) <= radius and
                abs(y - cy) <= radius and
                abs(z - cz) <= radius):
                if block.block_id in block_types:
                    count += 1
        
        return count
    
    def get_distance_to_nearest(
        self,
        world_snapshot: WorldSnapshot,
        block_types: set
    ) -> Optional[float]:
        """
        Get distance to nearest block of given types.
        
        Args:
            world_snapshot: World data
            block_types: Block types to find
            
        Returns:
            Distance to nearest, or None if not found
        """
        import math
        
        center = world_snapshot.center
        min_dist = float('inf')
        
        for (x, y, z), block in world_snapshot.blocks.items():
            if block.block_id in block_types:
                dist = math.sqrt(
                    (x - center.x) ** 2 +
                    (y - center.y) ** 2 +
                    (z - center.z) ** 2
                )
                min_dist = min(min_dist, dist)
        
        return min_dist if min_dist < float('inf') else None
    
    def get_entities_by_type(
        self,
        world_snapshot: WorldSnapshot,
        entity_type: str
    ) -> List[Entity]:
        """
        Get entities of a specific type.
        
        Args:
            world_snapshot: World data
            entity_type: Entity type ID (e.g., "minecraft:shulker")
            
        Returns:
            List of matching entities
        """
        return [e for e in world_snapshot.entities if e.entity_type == entity_type]
    
    def get_features_for_navigation(
        self,
        world_snapshot: WorldSnapshot
    ) -> Dict[str, Any]:
        """
        Extract navigation-relevant features from world snapshot.
        
        Returns features useful for pathfinding and decision making:
        - Ground level estimate
        - Nearby void gaps
        - Structure presence
        
        Args:
            world_snapshot: World data
            
        Returns:
            Dictionary of navigation features
        """
        center = world_snapshot.center
        
        # Find ground level at current position
        ground_y = self._estimate_ground_level(world_snapshot, center)
        
        # Count End structure blocks nearby
        end_structure_count = self.count_blocks_in_region(
            world_snapshot,
            (int(center.x), int(center.y), int(center.z)),
            32,
            self.END_BLOCKS
        )
        
        # Detect potential void gaps
        void_gaps = self._detect_void_gaps(world_snapshot)
        
        # Detect shulkers (danger)
        shulkers = self.get_entities_by_type(world_snapshot, "minecraft:shulker")
        
        return {
            "ground_y": ground_y,
            "end_structure_blocks": end_structure_count,
            "void_gap_count": len(void_gaps),
            "shulker_count": len(shulkers),
            "has_city_nearby": end_structure_count > 50,
        }
    
    def _estimate_ground_level(
        self,
        world_snapshot: WorldSnapshot,
        pos: Position
    ) -> int:
        """Estimate ground level at a position."""
        x, z = int(pos.x), int(pos.z)
        
        # Scan down from current Y
        for y in range(int(pos.y), 0, -1):
            block = world_snapshot.blocks.get((x, y, z))
            if block and block.block_id != "minecraft:air":
                return y + 1
        
        return 0  # Void
    
    def _detect_void_gaps(
        self,
        world_snapshot: WorldSnapshot
    ) -> List[Tuple[int, int]]:
        """Detect void gaps (areas with no ground)."""
        gaps = []
        center = world_snapshot.center
        
        # Sample points in a grid
        for dx in range(-32, 33, 8):
            for dz in range(-32, 33, 8):
                x, z = int(center.x + dx), int(center.z + dz)
                
                # Check if there's any ground
                has_ground = False
                for y in range(256, 0, -1):
                    block = world_snapshot.blocks.get((x, y, z))
                    if block and block.block_id != "minecraft:air":
                        has_ground = True
                        break
                
                if not has_ground:
                    gaps.append((x, z))
        
        return gaps
