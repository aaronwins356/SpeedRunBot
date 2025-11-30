"""
navigation.py - Pathfinding and movement primitives for Elytra Finder Bot.

This module provides navigation functionality:
- Basic movement (walk, jump, sneak)
- Pathfinding (A* over terrain)
- Safe traversal (avoiding void)
- Bridging and towering
"""

import time
import math
import logging
from typing import Optional, List, Tuple, Set
from dataclasses import dataclass
from heapq import heappush, heappop

from integration.mc_client import MinecraftClient, Position, Block

logger = logging.getLogger(__name__)


@dataclass
class PathNode:
    """A node in the pathfinding graph."""
    x: int
    y: int
    z: int
    g_cost: float = 0.0  # Cost from start
    h_cost: float = 0.0  # Heuristic cost to goal
    parent: Optional['PathNode'] = None
    
    @property
    def f_cost(self) -> float:
        """Total estimated cost."""
        return self.g_cost + self.h_cost
    
    def __lt__(self, other: 'PathNode') -> bool:
        return self.f_cost < other.f_cost
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PathNode):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z
    
    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z))


class Navigator:
    """
    Navigation system for the Minecraft bot.
    
    Provides pathfinding and movement primitives optimized for
    End navigation where void avoidance is critical.
    """
    
    # Block IDs considered walkable (air above, solid below)
    SOLID_BLOCKS = {
        "minecraft:stone", "minecraft:end_stone", "minecraft:purpur_block",
        "minecraft:purpur_pillar", "minecraft:end_stone_bricks",
        "minecraft:obsidian", "minecraft:dirt", "minecraft:grass_block",
        "minecraft:cobblestone", "minecraft:deepslate",
    }
    
    # Dangerous blocks
    DANGEROUS_BLOCKS = {
        "minecraft:lava", "minecraft:fire", "minecraft:cactus",
    }
    
    def __init__(self, client: MinecraftClient):
        """
        Initialize the navigator.
        
        Args:
            client: Minecraft client for movement commands
        """
        self.client = client
        
        # Navigation settings
        self.move_speed = 0.1
        self.path_timeout = 30.0  # seconds
        self.max_fall_distance = 3
        self.void_y_threshold = 0  # Y level below which is void
    
    def walk_to(
        self,
        target: Position,
        timeout: float = 30.0,
        tolerance: float = 2.0
    ) -> bool:
        """
        Walk to a target position.
        
        Args:
            target: Target position
            timeout: Maximum time to spend walking
            tolerance: How close to get (in blocks)
            
        Returns:
            True if reached target, False if failed/timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            pos = self.client.get_position()
            if pos is None:
                return False
            
            # Check if close enough
            distance = pos.distance_to(target)
            if distance < tolerance:
                return True
            
            # Calculate direction
            dx = target.x - pos.x
            dz = target.z - pos.z
            
            # Normalize
            dist_xz = math.sqrt(dx * dx + dz * dz)
            if dist_xz > 0:
                dx /= dist_xz
                dz /= dist_xz
            
            # Look at target
            self.client.look_at(target.x, target.y, target.z)
            
            # Move forward
            self.client.move(forward=1.0)
            
            # Small delay
            time.sleep(0.05)
        
        return False
    
    def safe_travel_to(
        self,
        target: Position,
        max_void_risk: float = 0.2
    ) -> bool:
        """
        Travel to target while avoiding void.
        
        Uses pathfinding to find a safe route, potentially
        including bridging over gaps.
        
        Args:
            target: Target position
            max_void_risk: Maximum acceptable void risk (0-1)
            
        Returns:
            True if reached target safely
        """
        pos = self.client.get_position()
        if pos is None:
            return False
        
        # Try to find a path
        path = self.find_path(
            (int(pos.x), int(pos.y), int(pos.z)),
            (int(target.x), int(target.y), int(target.z))
        )
        
        if path:
            # Follow the path
            return self.follow_path(path)
        else:
            # No path found, try direct approach with caution
            logger.warning("No path found, attempting cautious direct travel")
            return self._cautious_travel(target, max_void_risk)
    
    def find_path(
        self,
        start: Tuple[int, int, int],
        goal: Tuple[int, int, int],
        max_iterations: int = 1000
    ) -> Optional[List[Tuple[int, int, int]]]:
        """
        Find a path using A* algorithm.
        
        Args:
            start: Starting position (x, y, z)
            goal: Goal position (x, y, z)
            max_iterations: Maximum pathfinding iterations
            
        Returns:
            List of positions forming path, or None if no path found
        """
        start_node = PathNode(start[0], start[1], start[2])
        goal_node = PathNode(goal[0], goal[1], goal[2])
        
        start_node.h_cost = self._heuristic(start_node, goal_node)
        
        open_set: List[PathNode] = [start_node]
        closed_set: Set[Tuple[int, int, int]] = set()
        
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # Get node with lowest f_cost
            current = heappop(open_set)
            
            # Check if goal reached
            if current.x == goal_node.x and current.z == goal_node.z:
                # Reconstruct path
                return self._reconstruct_path(current)
            
            closed_set.add((current.x, current.y, current.z))
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                if (neighbor.x, neighbor.y, neighbor.z) in closed_set:
                    continue
                
                # Calculate costs
                move_cost = self._movement_cost(current, neighbor)
                new_g_cost = current.g_cost + move_cost
                
                # Check if this is a better path
                existing = self._find_in_open(open_set, neighbor)
                if existing and new_g_cost >= existing.g_cost:
                    continue
                
                neighbor.g_cost = new_g_cost
                neighbor.h_cost = self._heuristic(neighbor, goal_node)
                neighbor.parent = current
                
                if not existing:
                    heappush(open_set, neighbor)
        
        return None
    
    def follow_path(
        self,
        path: List[Tuple[int, int, int]],
        tolerance: float = 0.5
    ) -> bool:
        """
        Follow a pre-computed path.
        
        Args:
            path: List of positions to visit
            tolerance: How close to get to each waypoint
            
        Returns:
            True if successfully followed path
        """
        for waypoint in path:
            target = Position(waypoint[0] + 0.5, waypoint[1], waypoint[2] + 0.5)
            
            if not self.walk_to(target, timeout=10.0, tolerance=tolerance):
                logger.warning(f"Failed to reach waypoint {waypoint}")
                return False
        
        return True
    
    def bridge_over_gap(
        self,
        target: Position,
        max_gap: int = 10
    ) -> bool:
        """
        Bridge over a gap to reach target.
        
        Args:
            target: Target position to bridge to
            max_gap: Maximum gap size to bridge
            
        Returns:
            True if successfully bridged
        """
        pos = self.client.get_position()
        if pos is None:
            return False
        
        # Calculate bridging direction
        dx = target.x - pos.x
        dz = target.z - pos.z
        dist = math.sqrt(dx * dx + dz * dz)
        
        if dist > max_gap:
            logger.warning(f"Gap too large to bridge: {dist:.1f} blocks")
            return False
        
        # Normalize direction
        if dist > 0:
            dx /= dist
            dz /= dist
        
        # Sneak to edge
        self.client.move(sneak=True)
        
        # Bridge one block at a time
        for i in range(int(dist) + 1):
            # Look down and back
            self.client.look_at(pos.x - dx, pos.y - 1, pos.z - dz)
            
            # Place block
            self.client.place_block(
                int(pos.x + dx * i),
                int(pos.y - 1),
                int(pos.z + dz * i),
                face="top"
            )
            
            # Walk forward slightly
            self.client.move(forward=0.3, sneak=True)
            time.sleep(0.3)
        
        return True
    
    def tower_up(
        self,
        target_y: int,
        max_height: int = 20
    ) -> bool:
        """
        Build up to a target Y level.
        
        Args:
            target_y: Target Y coordinate
            max_height: Maximum blocks to place
            
        Returns:
            True if successfully towered
        """
        pos = self.client.get_position()
        if pos is None:
            return False
        
        blocks_to_place = min(int(target_y - pos.y), max_height)
        
        if blocks_to_place <= 0:
            return True
        
        for _ in range(blocks_to_place):
            # Jump
            self.client.move(jump=True)
            time.sleep(0.1)
            
            # Place block under
            self.client.look_at(pos.x, pos.y - 1, pos.z)
            self.client.place_block(int(pos.x), int(pos.y - 1), int(pos.z), face="top")
            time.sleep(0.2)
        
        return True
    
    def descend_safely(
        self,
        target_y: int,
        max_fall: int = 3
    ) -> bool:
        """
        Safely descend to a target Y level.
        
        Args:
            target_y: Target Y coordinate
            max_fall: Maximum safe fall distance
            
        Returns:
            True if successfully descended
        """
        pos = self.client.get_position()
        if pos is None:
            return False
        
        while pos.y > target_y:
            # Check if safe to drop
            fall_distance = self._get_fall_distance(pos)
            
            if fall_distance <= max_fall:
                # Safe to drop
                self.client.move(forward=0.3)
                time.sleep(0.2)
            else:
                # Need to descend carefully (place blocks, use ladder, etc.)
                logger.warning(f"Unsafe fall distance: {fall_distance}")
                return False
            
            pos = self.client.get_position()
            if pos is None:
                return False
        
        return True
    
    def _cautious_travel(
        self,
        target: Position,
        max_void_risk: float
    ) -> bool:
        """Travel cautiously when no path is found."""
        pos = self.client.get_position()
        if pos is None:
            return False
        
        # Take small steps, checking ground ahead
        max_steps = 100
        
        for _ in range(max_steps):
            pos = self.client.get_position()
            if pos is None:
                return False
            
            if pos.distance_to(target) < 2.0:
                return True
            
            # Check if ground ahead is safe
            ahead_x = int(pos.x + math.copysign(1, target.x - pos.x))
            ahead_z = int(pos.z + math.copysign(1, target.z - pos.z))
            
            ground_block = self.client.get_block_at(ahead_x, int(pos.y - 1), ahead_z)
            
            if ground_block is None or ground_block.block_id == "minecraft:air":
                # Void risk
                logger.warning("Void ahead, stopping")
                return False
            
            # Safe to move
            self.client.look_at(target.x, target.y, target.z)
            self.client.move(forward=0.5)
            time.sleep(0.1)
        
        return False
    
    def _get_neighbors(self, node: PathNode) -> List[PathNode]:
        """Get walkable neighbors of a node."""
        neighbors = []
        
        # 8 horizontal directions + up/down
        directions = [
            (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1),
            (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
            (0, 1, 0), (0, -1, 0)
        ]
        
        for dx, dy, dz in directions:
            nx, ny, nz = node.x + dx, node.y + dy, node.z + dz
            
            if self._is_walkable(nx, ny, nz):
                neighbors.append(PathNode(nx, ny, nz))
        
        return neighbors
    
    def _is_walkable(self, x: int, y: int, z: int) -> bool:
        """Check if a position is walkable."""
        # Above void?
        if y <= self.void_y_threshold:
            return False
        
        # Get block at position and below
        block_at = self.client.get_block_at(x, y, z)
        block_below = self.client.get_block_at(x, y - 1, z)
        block_above = self.client.get_block_at(x, y + 1, z)
        
        # Position should be air (can stand there)
        if block_at and block_at.block_id != "minecraft:air":
            return False
        
        # Above head should be air (can fit)
        if block_above and block_above.block_id != "minecraft:air":
            return False
        
        # Below should be solid (can stand on)
        if block_below is None or block_below.block_id == "minecraft:air":
            return False
        
        # Check for dangerous blocks
        if block_below and block_below.block_id in self.DANGEROUS_BLOCKS:
            return False
        
        return True
    
    def _heuristic(self, a: PathNode, b: PathNode) -> float:
        """Calculate heuristic distance (Manhattan + vertical cost)."""
        dx = abs(a.x - b.x)
        dy = abs(a.y - b.y)
        dz = abs(a.z - b.z)
        
        # Vertical movement is more expensive
        return dx + dz + dy * 2
    
    def _movement_cost(self, current: PathNode, neighbor: PathNode) -> float:
        """Calculate movement cost between nodes."""
        dx = abs(current.x - neighbor.x)
        dy = abs(current.y - neighbor.y)
        dz = abs(current.z - neighbor.z)
        
        # Diagonal movement costs more
        if dx + dz == 2:
            base_cost = 1.414
        else:
            base_cost = 1.0
        
        # Vertical movement costs more
        if dy > 0:
            base_cost += 0.5
        
        return base_cost
    
    def _reconstruct_path(
        self,
        node: PathNode
    ) -> List[Tuple[int, int, int]]:
        """Reconstruct path from goal node."""
        path = []
        current = node
        
        while current is not None:
            path.append((current.x, current.y, current.z))
            current = current.parent
        
        path.reverse()
        return path
    
    def _find_in_open(
        self,
        open_set: List[PathNode],
        target: PathNode
    ) -> Optional[PathNode]:
        """Find a node in the open set."""
        for node in open_set:
            if node.x == target.x and node.y == target.y and node.z == target.z:
                return node
        return None
    
    def _get_fall_distance(self, pos: Position) -> int:
        """Get distance to ground from position."""
        for dy in range(20):
            block = self.client.get_block_at(int(pos.x), int(pos.y - dy - 1), int(pos.z))
            if block and block.block_id != "minecraft:air":
                return dy
        return 20  # Max checked distance
