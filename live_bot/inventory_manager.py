"""
inventory_manager.py - Container parsing and Elytra detection.

This module provides inventory and container management:
- Chest content parsing
- Elytra detection
- Item movement and equipping
"""

import json
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from integration.mc_client import MinecraftClient, Item

logger = logging.getLogger(__name__)


@dataclass
class ElytraFindRecord:
    """Record of an Elytra find."""
    timestamp: str
    dimension: str
    x: float
    y: float
    z: float
    elytra_found: bool
    additional_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "timestamp": self.timestamp,
            "dimension": self.dimension,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "elytra_found": self.elytra_found,
        }
        if self.additional_info:
            result.update(self.additional_info)
        return result


class InventoryManager:
    """
    Inventory and container management for the Elytra Finder Bot.
    
    Handles:
    - Parsing container contents
    - Detecting Elytra items
    - Moving items between slots
    - Equipping Elytra as chestplate
    - Logging find results
    """
    
    # Elytra item identifiers
    ELYTRA_IDS = {
        "minecraft:elytra",
        "elytra",
    }
    
    # Armor slot indices in player inventory
    HELMET_SLOT = 39
    CHESTPLATE_SLOT = 38
    LEGGINGS_SLOT = 37
    BOOTS_SLOT = 36
    
    def __init__(
        self,
        client: MinecraftClient,
        log_file: str = "elytra_finds.jsonl"
    ):
        """
        Initialize the inventory manager.
        
        Args:
            client: Minecraft client for inventory operations
            log_file: File to log Elytra finds
        """
        self.client = client
        self.log_file = log_file
    
    def has_elytra(
        self,
        container_items: List[Optional[Item]]
    ) -> bool:
        """
        Check if container contains an Elytra.
        
        Args:
            container_items: List of items from container
            
        Returns:
            True if Elytra is present
        """
        for item in container_items:
            if item is None:
                continue
            
            if self._is_elytra(item):
                return True
        
        return False
    
    def get_elytra_slots(
        self,
        container_items: List[Optional[Item]]
    ) -> List[int]:
        """
        Get slot indices containing Elytra.
        
        Args:
            container_items: List of items from container
            
        Returns:
            List of slot indices with Elytra
        """
        slots = []
        
        for i, item in enumerate(container_items):
            if item is None:
                continue
            
            if self._is_elytra(item):
                slots.append(i)
        
        return slots
    
    def _is_elytra(self, item: Item) -> bool:
        """Check if an item is an Elytra."""
        if item is None:
            return False
        
        item_id = item.item_id.lower()
        
        # Check against known Elytra IDs
        if item_id in self.ELYTRA_IDS:
            return True
        
        # Also check for partial match
        if "elytra" in item_id:
            return True
        
        return False
    
    def take_elytra_from_container(
        self,
        slot: int
    ) -> bool:
        """
        Take Elytra from container and put in inventory.
        
        Args:
            slot: Slot index in container
            
        Returns:
            True if successful
        """
        if not self.client.is_connected():
            return False
        
        # This would use click packets to move the item
        # For now, log the action
        logger.info(f"Would take Elytra from container slot {slot}")
        
        # TODO: Implement actual item transfer
        # 1. Click on container slot (pick up item)
        # 2. Click on inventory slot (place item)
        
        return True
    
    def equip_elytra(
        self,
        inventory_slot: int
    ) -> bool:
        """
        Equip Elytra from inventory slot to chestplate slot.
        
        Args:
            inventory_slot: Slot index in inventory
            
        Returns:
            True if successful
        """
        if not self.client.is_connected():
            return False
        
        logger.info(f"Would equip Elytra from slot {inventory_slot} "
                   f"to chestplate slot {self.CHESTPLATE_SLOT}")
        
        # TODO: Implement actual equipping
        # 1. Click on inventory slot
        # 2. Click on armor slot
        
        return True
    
    def has_elytra_equipped(self) -> bool:
        """
        Check if player has Elytra equipped.
        
        Returns:
            True if Elytra is in chestplate slot
        """
        inventory = self.client.get_inventory()
        
        if len(inventory) <= self.CHESTPLATE_SLOT:
            return False
        
        chestplate = inventory[self.CHESTPLATE_SLOT]
        
        if chestplate is None:
            return False
        
        return self._is_elytra(chestplate)
    
    def get_elytra_durability(self) -> Optional[int]:
        """
        Get durability of equipped Elytra.
        
        Returns:
            Durability remaining, or None if not equipped
        """
        if not self.has_elytra_equipped():
            return None
        
        inventory = self.client.get_inventory()
        chestplate = inventory[self.CHESTPLATE_SLOT]
        
        if chestplate is None or chestplate.nbt is None:
            return None
        
        # Elytra has max durability of 432
        damage = chestplate.nbt.get("Damage", 0)
        return 432 - damage
    
    def log_elytra_find(
        self,
        elytra_found: bool,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an Elytra find to the log file.
        
        Args:
            elytra_found: Whether Elytra was found
            additional_info: Additional information to include
        """
        pos = self.client.get_position()
        
        record = ElytraFindRecord(
            timestamp=datetime.now().isoformat(),
            dimension=self.client.get_dimension(),
            x=pos.x if pos else 0,
            y=pos.y if pos else 0,
            z=pos.z if pos else 0,
            elytra_found=elytra_found,
            additional_info=additional_info
        )
        
        self._append_to_log(record)
    
    def _append_to_log(self, record: ElytraFindRecord) -> None:
        """Append a record to the log file."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(record.to_dict()) + "\n")
            logger.info(f"Logged find to {self.log_file}")
        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")
    
    def get_inventory_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the player's inventory.
        
        Returns:
            Dictionary with inventory summary
        """
        inventory = self.client.get_inventory()
        
        summary = {
            "total_slots": len(inventory),
            "empty_slots": sum(1 for item in inventory if item is None),
            "has_elytra": False,
            "elytra_durability": None,
            "item_counts": {}
        }
        
        for item in inventory:
            if item is None:
                continue
            
            if self._is_elytra(item):
                summary["has_elytra"] = True
            
            item_type = item.item_id
            if item_type not in summary["item_counts"]:
                summary["item_counts"][item_type] = 0
            summary["item_counts"][item_type] += item.count
        
        if summary["has_elytra"]:
            summary["elytra_durability"] = self.get_elytra_durability()
        
        return summary
    
    def find_empty_slot(self) -> Optional[int]:
        """
        Find an empty inventory slot.
        
        Returns:
            Slot index or None if full
        """
        inventory = self.client.get_inventory()
        
        # Hotbar slots (0-8) are preferred
        for i in range(9):
            if i < len(inventory) and inventory[i] is None:
                return i
        
        # Then main inventory (9-35)
        for i in range(9, 36):
            if i < len(inventory) and inventory[i] is None:
                return i
        
        return None
    
    def count_item(self, item_id: str) -> int:
        """
        Count total quantity of an item in inventory.
        
        Args:
            item_id: Item ID to count
            
        Returns:
            Total count
        """
        inventory = self.client.get_inventory()
        total = 0
        
        for item in inventory:
            if item is None:
                continue
            
            if item.item_id == item_id or item_id in item.item_id:
                total += item.count
        
        return total
