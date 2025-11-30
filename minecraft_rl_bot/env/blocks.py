"""
blocks.py - Block type definitions and properties for Minecraft RL environment.

This module defines all block types that can exist in the Minecraft world.
Each block has properties such as hardness, tool requirement, and drop items.

To extend:
- Add new block types to the BlockType enum
- Update BLOCK_PROPERTIES with properties for the new block
- Update any relevant crafting or interaction logic
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, List


class BlockType(IntEnum):
    """
    Enumeration of all block types in the environment.
    Integer values are used for efficient tensor representation.
    
    Categories:
    - 0-9: Basic natural blocks (air, dirt, stone, etc.)
    - 10-19: Ores and valuable blocks
    - 20-29: Crafted/processed blocks
    - 30-39: Nether blocks
    - 40-49: End blocks
    - 50-59: Liquids and special blocks
    - 60+: Entities and special markers
    """
    # Basic natural blocks
    AIR = 0
    DIRT = 1
    GRASS = 2
    STONE = 3
    COBBLESTONE = 4
    SAND = 5
    GRAVEL = 6
    WOOD_LOG = 7
    LEAVES = 8
    BEDROCK = 9
    
    # Ores and valuable blocks
    COAL_ORE = 10
    IRON_ORE = 11
    GOLD_ORE = 12
    DIAMOND_ORE = 13
    REDSTONE_ORE = 14
    LAPIS_ORE = 15
    EMERALD_ORE = 16
    
    # Crafted/processed blocks
    WOOD_PLANKS = 20
    CRAFTING_TABLE = 21
    FURNACE = 22
    CHEST = 23
    IRON_BLOCK = 24
    GOLD_BLOCK = 25
    DIAMOND_BLOCK = 26
    
    # Nether blocks
    NETHERRACK = 30
    NETHER_BRICK = 31
    SOUL_SAND = 32
    GLOWSTONE = 33
    NETHER_PORTAL = 34
    BLAZE_SPAWNER = 35
    NETHER_FORTRESS_BRICK = 36
    
    # End blocks
    END_STONE = 40
    END_PORTAL_FRAME = 41
    END_PORTAL = 42
    OBSIDIAN = 43
    
    # Liquids and special blocks
    WATER = 50
    LAVA = 51
    ICE = 52
    SNOW = 53
    
    # Entity markers (for observation tensor)
    ENTITY_HOSTILE = 60
    ENTITY_PASSIVE = 61
    ENTITY_ITEM = 62
    PLAYER = 63


@dataclass
class BlockProperties:
    """
    Properties for each block type.
    
    Attributes:
        hardness: Time in seconds to mine with bare hands (0 = instant, -1 = unbreakable)
        tool_required: Minimum tool type needed to mine ('none', 'wood', 'stone', 'iron', 'diamond')
        drops: List of (ItemType, count) tuples that drop when mined
        flammable: Whether the block can catch fire
        transparent: Whether light passes through
        solid: Whether entities can walk through
    """
    hardness: float
    tool_required: str
    drops: List[tuple]
    flammable: bool = False
    transparent: bool = False
    solid: bool = True


# Block properties dictionary
# Maps BlockType to BlockProperties
BLOCK_PROPERTIES = {
    BlockType.AIR: BlockProperties(0, 'none', [], transparent=True, solid=False),
    BlockType.DIRT: BlockProperties(0.5, 'none', [('dirt', 1)]),
    BlockType.GRASS: BlockProperties(0.6, 'none', [('dirt', 1)]),
    BlockType.STONE: BlockProperties(1.5, 'wood', [('cobblestone', 1)]),
    BlockType.COBBLESTONE: BlockProperties(2.0, 'wood', [('cobblestone', 1)]),
    BlockType.SAND: BlockProperties(0.5, 'none', [('sand', 1)]),
    BlockType.GRAVEL: BlockProperties(0.6, 'none', [('gravel', 1)]),
    BlockType.WOOD_LOG: BlockProperties(2.0, 'none', [('wood_log', 1)], flammable=True),
    BlockType.LEAVES: BlockProperties(0.2, 'none', [], flammable=True, transparent=True),
    BlockType.BEDROCK: BlockProperties(-1, 'none', []),  # Unbreakable
    
    BlockType.COAL_ORE: BlockProperties(3.0, 'wood', [('coal', 1)]),
    BlockType.IRON_ORE: BlockProperties(3.0, 'stone', [('iron_ore', 1)]),
    BlockType.GOLD_ORE: BlockProperties(3.0, 'iron', [('gold_ore', 1)]),
    BlockType.DIAMOND_ORE: BlockProperties(3.0, 'iron', [('diamond', 1)]),
    BlockType.REDSTONE_ORE: BlockProperties(3.0, 'iron', [('redstone', 4)]),
    BlockType.LAPIS_ORE: BlockProperties(3.0, 'stone', [('lapis', 4)]),
    BlockType.EMERALD_ORE: BlockProperties(3.0, 'iron', [('emerald', 1)]),
    
    BlockType.WOOD_PLANKS: BlockProperties(2.0, 'none', [('wood_planks', 1)], flammable=True),
    BlockType.CRAFTING_TABLE: BlockProperties(2.5, 'none', [('crafting_table', 1)], flammable=True),
    BlockType.FURNACE: BlockProperties(3.5, 'wood', [('furnace', 1)]),
    BlockType.CHEST: BlockProperties(2.5, 'none', [('chest', 1)], flammable=True),
    BlockType.IRON_BLOCK: BlockProperties(5.0, 'stone', [('iron_block', 1)]),
    BlockType.GOLD_BLOCK: BlockProperties(3.0, 'iron', [('gold_block', 1)]),
    BlockType.DIAMOND_BLOCK: BlockProperties(5.0, 'iron', [('diamond_block', 1)]),
    
    BlockType.NETHERRACK: BlockProperties(0.4, 'wood', [('netherrack', 1)]),
    BlockType.NETHER_BRICK: BlockProperties(2.0, 'wood', [('nether_brick', 1)]),
    BlockType.SOUL_SAND: BlockProperties(0.5, 'none', [('soul_sand', 1)]),
    BlockType.GLOWSTONE: BlockProperties(0.3, 'none', [('glowstone_dust', 2)]),
    BlockType.NETHER_PORTAL: BlockProperties(-1, 'none', [], transparent=True, solid=False),
    BlockType.BLAZE_SPAWNER: BlockProperties(5.0, 'wood', []),
    BlockType.NETHER_FORTRESS_BRICK: BlockProperties(2.0, 'wood', [('nether_brick', 1)]),
    
    BlockType.END_STONE: BlockProperties(3.0, 'wood', [('end_stone', 1)]),
    BlockType.END_PORTAL_FRAME: BlockProperties(-1, 'none', []),
    BlockType.END_PORTAL: BlockProperties(-1, 'none', [], transparent=True, solid=False),
    BlockType.OBSIDIAN: BlockProperties(50.0, 'diamond', [('obsidian', 1)]),
    
    BlockType.WATER: BlockProperties(0, 'none', [], transparent=True, solid=False),
    BlockType.LAVA: BlockProperties(0, 'none', [], transparent=True, solid=False),
    BlockType.ICE: BlockProperties(0.5, 'none', []),
    BlockType.SNOW: BlockProperties(0.1, 'none', [('snowball', 1)]),
    
    BlockType.ENTITY_HOSTILE: BlockProperties(0, 'none', [], solid=False),
    BlockType.ENTITY_PASSIVE: BlockProperties(0, 'none', [], solid=False),
    BlockType.ENTITY_ITEM: BlockProperties(0, 'none', [], solid=False),
    BlockType.PLAYER: BlockProperties(0, 'none', [], solid=False),
}


def get_block_hardness(block_type: BlockType) -> float:
    """Get the hardness of a block type."""
    return BLOCK_PROPERTIES.get(block_type, BLOCK_PROPERTIES[BlockType.STONE]).hardness


def is_solid(block_type: BlockType) -> bool:
    """Check if a block is solid (entities cannot pass through)."""
    return BLOCK_PROPERTIES.get(block_type, BLOCK_PROPERTIES[BlockType.STONE]).solid


def is_transparent(block_type: BlockType) -> bool:
    """Check if a block is transparent (light passes through)."""
    return BLOCK_PROPERTIES.get(block_type, BLOCK_PROPERTIES[BlockType.AIR]).transparent


def can_mine_with_tool(block_type: BlockType, tool_level: str) -> bool:
    """
    Check if a block can be mined with the given tool level.
    
    Tool levels (in order): 'none', 'wood', 'stone', 'iron', 'diamond'
    """
    tool_hierarchy = ['none', 'wood', 'stone', 'iron', 'diamond']
    required = BLOCK_PROPERTIES.get(block_type, BLOCK_PROPERTIES[BlockType.STONE]).tool_required
    
    if required not in tool_hierarchy or tool_level not in tool_hierarchy:
        return False
    
    return tool_hierarchy.index(tool_level) >= tool_hierarchy.index(required)
