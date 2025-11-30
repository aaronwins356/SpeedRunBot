"""
end_city_scanner.py - Detection of End Cities and End Ships.

This module provides algorithms for detecting:
- End City structures (purpur + end stone brick patterns)
- End Ships (unique ship hull shape with end rods + dragon head)
- Ship chest locations for Elytra retrieval
"""

import logging
from typing import List, Optional, Tuple, Dict, Set
from dataclasses import dataclass
import math

from integration.mc_client import MinecraftClient, Position
from .perception import PerceptionModule, WorldSnapshot

logger = logging.getLogger(__name__)


@dataclass
class CityCandidate:
    """A potential End City location."""
    x: int
    y: int
    z: int
    confidence: float  # 0.0 to 1.0
    block_count: int
    features: Dict[str, int]
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CityCandidate):
            return False
        # Cities within 32 blocks are considered the same
        return (abs(self.x - other.x) < 32 and
                abs(self.z - other.z) < 32)
    
    def __hash__(self) -> int:
        # Hash by approximate chunk position
        return hash((self.x // 32, self.z // 32))


@dataclass
class ShipCandidate:
    """A potential End Ship location."""
    x: int
    y: int
    z: int
    confidence: float
    has_dragon_head: bool
    has_brewing_stand: bool
    has_chest: bool
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShipCandidate):
            return False
        return (abs(self.x - other.x) < 16 and
                abs(self.y - other.y) < 10 and
                abs(self.z - other.z) < 16)
    
    def __hash__(self) -> int:
        return hash((self.x // 16, self.y // 10, self.z // 16))


class EndCityScanner:
    """
    Scanner for detecting End Cities and End Ships.
    
    Uses block pattern recognition to identify:
    - End City towers (purpur + end stone brick)
    - End Ships (purpur hull + end rods + dragon head)
    
    The scanner analyzes world snapshots from the perception module
    and returns candidate locations with confidence scores.
    """
    
    # Block types for End City detection
    CITY_BLOCKS = {
        "minecraft:purpur_block",
        "minecraft:purpur_pillar",
        "minecraft:purpur_stairs",
        "minecraft:purpur_slab",
        "minecraft:end_stone_bricks",
        "minecraft:end_rod",
    }
    
    # Block types specific to End Ships
    SHIP_BLOCKS = {
        "minecraft:purpur_block",
        "minecraft:purpur_stairs",
        "minecraft:purpur_slab",
    }
    
    # Ship signature blocks (unique to ships)
    SHIP_SIGNATURE = {
        "minecraft:dragon_head",
        "minecraft:brewing_stand",
        "minecraft:chest",
    }
    
    # Minimum blocks to consider a valid city
    MIN_CITY_BLOCKS = 20
    
    # Minimum confidence to report a candidate
    MIN_CONFIDENCE = 0.3
    
    def __init__(
        self,
        client: MinecraftClient,
        perception: PerceptionModule
    ):
        """
        Initialize the scanner.
        
        Args:
            client: Minecraft client for queries
            perception: Perception module for world data
        """
        self.client = client
        self.perception = perception
    
    def find_potential_end_cities(
        self,
        world_snapshot: WorldSnapshot
    ) -> List[CityCandidate]:
        """
        Find potential End City locations in a world snapshot.
        
        Uses clustering of End City blocks to identify
        potential structures.
        
        Args:
            world_snapshot: World data to analyze
            
        Returns:
            List of CityCandidate objects sorted by confidence
        """
        # Find all city-related blocks
        city_block_positions = self.perception.find_blocks_by_type(
            world_snapshot,
            self.CITY_BLOCKS
        )
        
        if not city_block_positions:
            return []
        
        logger.debug(f"Found {len(city_block_positions)} city-related blocks")
        
        # Cluster blocks into potential cities
        clusters = self._cluster_positions(city_block_positions, cluster_distance=32)
        
        candidates = []
        
        for cluster in clusters:
            if len(cluster) < self.MIN_CITY_BLOCKS:
                continue
            
            # Calculate cluster center
            center_x = sum(p[0] for p in cluster) // len(cluster)
            center_y = sum(p[1] for p in cluster) // len(cluster)
            center_z = sum(p[2] for p in cluster) // len(cluster)
            
            # Analyze cluster features
            features = self._analyze_cluster(cluster, world_snapshot)
            
            # Calculate confidence based on features
            confidence = self._calculate_city_confidence(features, len(cluster))
            
            if confidence >= self.MIN_CONFIDENCE:
                candidates.append(CityCandidate(
                    x=center_x,
                    y=center_y,
                    z=center_z,
                    confidence=confidence,
                    block_count=len(cluster),
                    features=features
                ))
        
        # Sort by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        
        logger.info(f"Found {len(candidates)} city candidates")
        
        return candidates
    
    def find_potential_ships(
        self,
        world_snapshot: WorldSnapshot,
        cities: List[CityCandidate]
    ) -> List[ShipCandidate]:
        """
        Find potential End Ships in a world snapshot.
        
        Ships are typically attached to End Cities via bridges,
        so we search near known city locations.
        
        Args:
            world_snapshot: World data to analyze
            cities: Known city candidates to search near
            
        Returns:
            List of ShipCandidate objects sorted by confidence
        """
        candidates = []
        
        # Search near each city
        for city in cities:
            ship_candidates = self._find_ships_near_city(
                world_snapshot,
                (city.x, city.y, city.z)
            )
            candidates.extend(ship_candidates)
        
        # Also do a general search for ship signatures
        general_ships = self._find_ships_by_signature(world_snapshot)
        candidates.extend(general_ships)
        
        # Deduplicate
        seen: Set[ShipCandidate] = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)
        
        # Sort by confidence
        unique_candidates.sort(key=lambda s: s.confidence, reverse=True)
        
        logger.info(f"Found {len(unique_candidates)} ship candidates")
        
        return unique_candidates
    
    def find_ship_chest_position(
        self,
        ship: ShipCandidate
    ) -> Optional[Tuple[int, int, int]]:
        """
        Find the chest position inside an End Ship.
        
        End Ship chests are in a consistent relative position
        within the ship structure.
        
        Args:
            ship: Ship candidate to search
            
        Returns:
            (x, y, z) of chest, or None if not found
        """
        # The chest in an End Ship is typically in the hull
        # Search in a small area around the ship center
        search_radius = 8
        
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-4, 5):
                for dz in range(-search_radius, search_radius + 1):
                    x, y, z = ship.x + dx, ship.y + dy, ship.z + dz
                    block = self.client.get_block_at(x, y, z)
                    
                    if block and block.block_id == "minecraft:chest":
                        logger.info(f"Found ship chest at ({x}, {y}, {z})")
                        return (x, y, z)
        
        logger.warning("Could not find chest in ship")
        return None
    
    def _cluster_positions(
        self,
        positions: List[Tuple[int, int, int]],
        cluster_distance: int
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Cluster positions using simple distance-based clustering.
        
        Args:
            positions: List of (x, y, z) positions
            cluster_distance: Maximum distance between cluster members
            
        Returns:
            List of clusters, each a list of positions
        """
        if not positions:
            return []
        
        clusters: List[List[Tuple[int, int, int]]] = []
        assigned = [False] * len(positions)
        
        for i, pos in enumerate(positions):
            if assigned[i]:
                continue
            
            # Start new cluster
            cluster = [pos]
            assigned[i] = True
            
            # Find all positions within distance
            for j, other in enumerate(positions):
                if assigned[j]:
                    continue
                
                # Check distance to any cluster member
                for member in cluster:
                    dist = math.sqrt(
                        (pos[0] - member[0]) ** 2 +
                        (pos[1] - member[1]) ** 2 +
                        (pos[2] - member[2]) ** 2
                    )
                    if dist <= cluster_distance:
                        cluster.append(other)
                        assigned[j] = True
                        break
            
            clusters.append(cluster)
        
        return clusters
    
    def _analyze_cluster(
        self,
        cluster: List[Tuple[int, int, int]],
        world_snapshot: WorldSnapshot
    ) -> Dict[str, int]:
        """Analyze a cluster to count different block types."""
        features = {
            "purpur_blocks": 0,
            "end_stone_bricks": 0,
            "end_rods": 0,
            "stairs": 0,
            "height_span": 0,
        }
        
        min_y = min(p[1] for p in cluster)
        max_y = max(p[1] for p in cluster)
        features["height_span"] = max_y - min_y
        
        for pos in cluster:
            block = world_snapshot.blocks.get(pos)
            if block is None:
                continue
            
            if "purpur" in block.block_id:
                features["purpur_blocks"] += 1
            if "end_stone_bricks" in block.block_id:
                features["end_stone_bricks"] += 1
            if "end_rod" in block.block_id:
                features["end_rods"] += 1
            if "stairs" in block.block_id:
                features["stairs"] += 1
        
        return features
    
    def _calculate_city_confidence(
        self,
        features: Dict[str, int],
        block_count: int
    ) -> float:
        """
        Calculate confidence that a cluster is an End City.
        
        Args:
            features: Block type counts
            block_count: Total blocks in cluster
            
        Returns:
            Confidence score 0.0 to 1.0
        """
        confidence = 0.0
        
        # Base confidence from block count
        confidence += min(block_count / 100.0, 0.3)
        
        # Purpur blocks are essential
        if features["purpur_blocks"] > 10:
            confidence += 0.2
        
        # End stone bricks indicate proper structure
        if features["end_stone_bricks"] > 5:
            confidence += 0.15
        
        # End rods are common in cities
        if features["end_rods"] > 2:
            confidence += 0.1
        
        # Height span indicates tower structure
        if features["height_span"] > 20:
            confidence += 0.15
        
        # Stairs indicate detailed structure
        if features["stairs"] > 5:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _find_ships_near_city(
        self,
        world_snapshot: WorldSnapshot,
        city_center: Tuple[int, int, int]
    ) -> List[ShipCandidate]:
        """Find ships near a city center."""
        candidates = []
        cx, cy, cz = city_center
        
        # Ships are usually attached via bridges, search in a radius
        search_radius = 64
        
        # Look for ship signature blocks
        for pos, block in world_snapshot.blocks.items():
            if block.block_id not in self.SHIP_SIGNATURE:
                continue
            
            x, y, z = pos
            
            # Check if within search area
            dist = math.sqrt((x - cx) ** 2 + (z - cz) ** 2)
            if dist > search_radius:
                continue
            
            # Analyze around signature block
            ship = self._analyze_potential_ship(world_snapshot, pos)
            if ship:
                candidates.append(ship)
        
        return candidates
    
    def _find_ships_by_signature(
        self,
        world_snapshot: WorldSnapshot
    ) -> List[ShipCandidate]:
        """Find ships by looking for unique ship blocks."""
        candidates = []
        
        # Find dragon heads (unique to ships)
        for pos, block in world_snapshot.blocks.items():
            if block.block_id == "minecraft:dragon_head":
                ship = self._analyze_potential_ship(world_snapshot, pos)
                if ship:
                    ship.has_dragon_head = True
                    # Dragon head is a strong indicator
                    ship.confidence = min(ship.confidence + 0.3, 1.0)
                    candidates.append(ship)
        
        return candidates
    
    def _analyze_potential_ship(
        self,
        world_snapshot: WorldSnapshot,
        center: Tuple[int, int, int]
    ) -> Optional[ShipCandidate]:
        """Analyze area around a point to see if it's a ship."""
        cx, cy, cz = center
        
        # Count ship-related blocks nearby
        purpur_count = 0
        has_chest = False
        has_brewing_stand = False
        has_dragon_head = False
        
        search_radius = 10
        
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-5, 10):
                for dz in range(-search_radius, search_radius + 1):
                    block = world_snapshot.blocks.get((cx + dx, cy + dy, cz + dz))
                    if block is None:
                        continue
                    
                    if "purpur" in block.block_id:
                        purpur_count += 1
                    if block.block_id == "minecraft:chest":
                        has_chest = True
                    if block.block_id == "minecraft:brewing_stand":
                        has_brewing_stand = True
                    if block.block_id == "minecraft:dragon_head":
                        has_dragon_head = True
        
        # Need significant purpur presence
        if purpur_count < 30:
            return None
        
        # Calculate confidence
        confidence = 0.3  # Base confidence
        
        if has_chest:
            confidence += 0.2
        if has_brewing_stand:
            confidence += 0.15
        if has_dragon_head:
            confidence += 0.25
        if purpur_count > 50:
            confidence += 0.1
        
        if confidence < self.MIN_CONFIDENCE:
            return None
        
        return ShipCandidate(
            x=cx,
            y=cy,
            z=cz,
            confidence=confidence,
            has_dragon_head=has_dragon_head,
            has_brewing_stand=has_brewing_stand,
            has_chest=has_chest
        )
