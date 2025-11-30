"""
actions.py - Action definitions for Minecraft RL environment.

This module defines all possible actions the agent can take.
Supports both discrete and continuous action spaces.

The action system is designed to be extensible:
- Discrete actions are indexed for use with classification networks
- Continuous actions can be parameterized for fine-grained control

To extend:
- Add new actions to the appropriate ActionType enum
- Update action_to_index and index_to_action mappings
- Implement action effects in core_env.py
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


class MovementAction(IntEnum):
    """
    Movement actions for the agent.
    These control the agent's position in the world.
    """
    NONE = 0
    FORWARD = 1
    BACKWARD = 2
    LEFT = 3
    RIGHT = 4
    JUMP = 5
    SNEAK = 6
    SPRINT = 7


class CameraAction(IntEnum):
    """
    Camera/look actions for the agent.
    These control where the agent is looking.
    
    In discrete mode: Fixed angle increments
    In continuous mode: Degrees of rotation
    """
    NONE = 0
    LOOK_UP = 1
    LOOK_DOWN = 2
    LOOK_LEFT = 3
    LOOK_RIGHT = 4


class InteractionAction(IntEnum):
    """
    Interaction actions for the agent.
    These control how the agent interacts with the world.
    """
    NONE = 0
    MINE = 1           # Break block in front
    PLACE = 2          # Place block from inventory
    USE = 3            # Use item/interact with block
    ATTACK = 4         # Attack entity in front
    CRAFT = 5          # Open crafting/craft item
    SMELT = 6          # Use furnace


class InventoryAction(IntEnum):
    """
    Inventory management actions.
    These control the agent's inventory state.
    """
    NONE = 0
    SELECT_SLOT_1 = 1
    SELECT_SLOT_2 = 2
    SELECT_SLOT_3 = 3
    SELECT_SLOT_4 = 4
    SELECT_SLOT_5 = 5
    SELECT_SLOT_6 = 6
    SELECT_SLOT_7 = 7
    SELECT_SLOT_8 = 8
    SELECT_SLOT_9 = 9
    DROP_ITEM = 10
    SWAP_HANDS = 11


@dataclass
class DiscreteAction:
    """
    A complete discrete action combining all action types.
    
    The agent outputs indices for each action type, which are
    then decoded into actual game actions.
    """
    movement: MovementAction = MovementAction.NONE
    camera: CameraAction = CameraAction.NONE
    interaction: InteractionAction = InteractionAction.NONE
    inventory: InventoryAction = InventoryAction.NONE
    
    def to_dict(self) -> Dict[str, int]:
        """Convert action to dictionary format."""
        return {
            'movement': self.movement.value,
            'camera': self.camera.value,
            'interaction': self.interaction.value,
            'inventory': self.inventory.value
        }
    
    @classmethod
    def from_dict(cls, action_dict: Dict[str, int]) -> 'DiscreteAction':
        """Create action from dictionary format."""
        return cls(
            movement=MovementAction(action_dict.get('movement', 0)),
            camera=CameraAction(action_dict.get('camera', 0)),
            interaction=InteractionAction(action_dict.get('interaction', 0)),
            inventory=InventoryAction(action_dict.get('inventory', 0))
        )


@dataclass
class ContinuousAction:
    """
    A continuous action for fine-grained control.
    
    This is used for advanced training where the agent needs
    more precise control over movement and camera.
    
    Attributes:
        move_forward: Forward/backward velocity [-1, 1]
        move_strafe: Left/right strafe velocity [-1, 1]
        camera_pitch: Vertical camera rotation in degrees [-180, 180]
        camera_yaw: Horizontal camera rotation in degrees [-180, 180]
        jump: Whether to jump [0, 1]
        interaction: Interaction action index
        inventory_slot: Selected inventory slot [0-8]
    """
    move_forward: float = 0.0
    move_strafe: float = 0.0
    camera_pitch: float = 0.0
    camera_yaw: float = 0.0
    jump: float = 0.0
    interaction: int = 0
    inventory_slot: int = 0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for network output."""
        return np.array([
            self.move_forward,
            self.move_strafe,
            self.camera_pitch,
            self.camera_yaw,
            self.jump,
            float(self.interaction),
            float(self.inventory_slot)
        ], dtype=np.float32)
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'ContinuousAction':
        """Create action from numpy array."""
        return cls(
            move_forward=float(np.clip(arr[0], -1, 1)),
            move_strafe=float(np.clip(arr[1], -1, 1)),
            camera_pitch=float(np.clip(arr[2], -180, 180)),
            camera_yaw=float(np.clip(arr[3], -180, 180)),
            jump=float(np.clip(arr[4], 0, 1)),
            interaction=int(np.clip(arr[5], 0, len(InteractionAction) - 1)),
            inventory_slot=int(np.clip(arr[6], 0, 8))
        )


# Action space dimensions for discrete actions
DISCRETE_ACTION_DIMS = {
    'movement': len(MovementAction),
    'camera': len(CameraAction),
    'interaction': len(InteractionAction),
    'inventory': len(InventoryAction)
}

# Total number of discrete actions (for flat action space)
TOTAL_DISCRETE_ACTIONS = sum(DISCRETE_ACTION_DIMS.values())

# Continuous action dimensions
CONTINUOUS_ACTION_DIM = 7


def create_action_space_info(continuous: bool = False) -> Dict:
    """
    Create action space information for the environment.
    
    Args:
        continuous: Whether to use continuous action space
        
    Returns:
        Dictionary with action space information
    """
    if continuous:
        return {
            'type': 'continuous',
            'dim': CONTINUOUS_ACTION_DIM,
            'low': np.array([-1, -1, -180, -180, 0, 0, 0], dtype=np.float32),
            'high': np.array([1, 1, 180, 180, 1, len(InteractionAction) - 1, 8], dtype=np.float32)
        }
    else:
        return {
            'type': 'discrete',
            'dims': DISCRETE_ACTION_DIMS,
            'total': TOTAL_DISCRETE_ACTIONS
        }


def sample_random_action(continuous: bool = False) -> DiscreteAction | ContinuousAction:
    """
    Sample a random action for exploration.
    
    Args:
        continuous: Whether to sample continuous action
        
    Returns:
        Random DiscreteAction or ContinuousAction
    """
    if continuous:
        return ContinuousAction(
            move_forward=np.random.uniform(-1, 1),
            move_strafe=np.random.uniform(-1, 1),
            camera_pitch=np.random.uniform(-30, 30),
            camera_yaw=np.random.uniform(-30, 30),
            jump=np.random.random(),
            interaction=np.random.randint(0, len(InteractionAction)),
            inventory_slot=np.random.randint(0, 9)
        )
    else:
        return DiscreteAction(
            movement=MovementAction(np.random.randint(0, len(MovementAction))),
            camera=CameraAction(np.random.randint(0, len(CameraAction))),
            interaction=InteractionAction(np.random.randint(0, len(InteractionAction))),
            inventory=InventoryAction(np.random.randint(0, len(InventoryAction)))
        )


# Camera rotation constants for discrete actions
DISCRETE_CAMERA_ANGLE = 15.0  # Degrees per discrete camera action

# Movement constants
WALK_SPEED = 4.317  # Blocks per second (Minecraft default)
SPRINT_SPEED = 5.612  # Blocks per second
SNEAK_SPEED = 1.31  # Blocks per second
