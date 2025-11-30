"""
controller.py - High-level bot logic and state machine for Elytra Finder Bot.

This module implements the main bot controller that:
- Manages bot state transitions
- Coordinates all bot subsystems
- Handles the Elytra search workflow
- Logs results and handles errors

States:
- CONNECT: Initial connection to server
- LOGIN_FLOW: Navigate server hub to correct world
- ENTER_END: Get to The End dimension
- SEARCH_FOR_CITY: Scout for End Cities
- PATH_TO_CITY: Navigate to a found city
- SEARCH_FOR_SHIP: Look for End Ship at city
- PATH_TO_SHIP: Navigate to the ship
- OPEN_SHIP_CHEST: Open the ship's chest
- CHECK_FOR_ELYTRA: Check for Elytra in chest
- LOG_RESULT: Log the finding
- MOVE_TO_NEXT_TARGET: Move to next search area
- IDLE: Waiting state
- STOP: Shutdown state
- ERROR: Error recovery state
"""

import time
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from datetime import datetime

from integration.mc_client import MinecraftClient, ClientConfig, Position
from .navigation import Navigator
from .perception import PerceptionModule
from .end_city_scanner import EndCityScanner, CityCandidate, ShipCandidate
from .inventory_manager import InventoryManager
from .login_flow import LoginFlow

logger = logging.getLogger(__name__)


class BotState(Enum):
    """Bot state machine states."""
    CONNECT = auto()
    LOGIN_FLOW = auto()
    ENTER_END = auto()
    SEARCH_FOR_CITY = auto()
    PATH_TO_CITY = auto()
    SEARCH_FOR_SHIP = auto()
    PATH_TO_SHIP = auto()
    OPEN_SHIP_CHEST = auto()
    CHECK_FOR_ELYTRA = auto()
    LOG_RESULT = auto()
    MOVE_TO_NEXT_TARGET = auto()
    IDLE = auto()
    STOP = auto()
    ERROR = auto()


@dataclass
class BotConfig:
    """Configuration for the bot controller."""
    # Server settings
    host: str = "play.lemoncloud.net"
    port: int = 25565
    username: str = ""
    
    # Login flow
    login_commands: List[str] = field(default_factory=lambda: ["/survival"])
    go_to_end_commands: List[str] = field(default_factory=lambda: ["/warp end"])
    command_delay: float = 2.0
    
    # Behavior settings
    max_runtime_minutes: int = 120
    log_interval_seconds: int = 5
    search_radius_blocks: int = 256
    max_void_risk: float = 0.2
    
    # Control mode
    control_mode: str = "scripted"  # "scripted" or "rl"
    
    # Safety
    dry_run: bool = False
    max_sessions_per_day: int = 10
    
    # Output
    elytra_log_file: str = "elytra_finds.jsonl"


@dataclass
class ElytraFind:
    """Record of an Elytra find."""
    timestamp: str
    dimension: str
    x: float
    y: float
    z: float
    elytra_found: bool
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "dimension": self.dimension,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "elytra_found": self.elytra_found,
            **self.additional_info
        }


class BotController:
    """
    High-level controller for the Elytra Finder Bot.
    
    This class orchestrates all bot behavior through a state machine,
    coordinating the various subsystems (navigation, perception, etc.)
    to search for and log Elytra locations.
    
    Usage:
        config = BotConfig(host="server.com", username="bot")
        controller = BotController(config)
        controller.run()
    """
    
    def __init__(self, config: BotConfig):
        """
        Initialize the bot controller.
        
        Args:
            config: Bot configuration
        """
        self.config = config
        
        # Create Minecraft client
        client_config = ClientConfig(
            host=config.host,
            port=config.port,
            username=config.username,
            dry_run=config.dry_run
        )
        self.client = MinecraftClient(client_config)
        
        # Create subsystems
        self.navigator = Navigator(self.client)
        self.perception = PerceptionModule(self.client)
        self.scanner = EndCityScanner(self.client, self.perception)
        self.inventory = InventoryManager(self.client)
        self.login_flow = LoginFlow(
            self.client,
            config.login_commands,
            config.go_to_end_commands,
            config.command_delay
        )
        
        # State machine
        self._state = BotState.CONNECT
        self._previous_state = None
        self._state_start_time = time.time()
        
        # Search state
        self._current_city: Optional[CityCandidate] = None
        self._current_ship: Optional[ShipCandidate] = None
        self._searched_cities: List[CityCandidate] = []
        self._elytra_finds: List[ElytraFind] = []
        
        # Runtime tracking
        self._start_time = time.time()
        self._last_log_time = time.time()
        
        # RL policy (optional, loaded if control_mode is "rl")
        self._policy = None
        if config.control_mode == "rl":
            self._load_policy()
        
        logger.info(f"BotController initialized (mode={config.control_mode})")
    
    def run(self) -> None:
        """
        Main bot loop.
        
        Runs the state machine until stopped or runtime limit reached.
        """
        logger.info("Starting Elytra Finder Bot...")
        logger.info(f"Target: {self.config.host}:{self.config.port}")
        logger.info(f"Max runtime: {self.config.max_runtime_minutes} minutes")
        
        try:
            while self._state != BotState.STOP:
                # Check runtime limit
                if self._check_runtime_limit():
                    logger.info("Runtime limit reached, stopping...")
                    self._state = BotState.STOP
                    break
                
                # Process current state
                self._process_state()
                
                # Update client
                self.client.update()
                
                # Periodic logging
                self._periodic_log()
                
                # Small delay to prevent busy-waiting
                time.sleep(0.05)
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self._shutdown()
    
    def _process_state(self) -> None:
        """Process the current state and transition if needed."""
        handlers = {
            BotState.CONNECT: self._handle_connect,
            BotState.LOGIN_FLOW: self._handle_login_flow,
            BotState.ENTER_END: self._handle_enter_end,
            BotState.SEARCH_FOR_CITY: self._handle_search_for_city,
            BotState.PATH_TO_CITY: self._handle_path_to_city,
            BotState.SEARCH_FOR_SHIP: self._handle_search_for_ship,
            BotState.PATH_TO_SHIP: self._handle_path_to_ship,
            BotState.OPEN_SHIP_CHEST: self._handle_open_ship_chest,
            BotState.CHECK_FOR_ELYTRA: self._handle_check_for_elytra,
            BotState.LOG_RESULT: self._handle_log_result,
            BotState.MOVE_TO_NEXT_TARGET: self._handle_move_to_next,
            BotState.IDLE: self._handle_idle,
            BotState.ERROR: self._handle_error,
        }
        
        handler = handlers.get(self._state)
        if handler:
            handler()
    
    def _transition_to(self, new_state: BotState) -> None:
        """Transition to a new state."""
        if new_state != self._state:
            self._previous_state = self._state
            self._state = new_state
            self._state_start_time = time.time()
            logger.info(f"State: {self._previous_state.name} -> {new_state.name}")
    
    def _handle_connect(self) -> None:
        """Handle CONNECT state."""
        logger.info("Connecting to server...")
        
        if self.client.connect():
            logger.info("Connected successfully")
            self._transition_to(BotState.LOGIN_FLOW)
        else:
            logger.error("Failed to connect")
            self._transition_to(BotState.ERROR)
    
    def _handle_login_flow(self) -> None:
        """Handle LOGIN_FLOW state."""
        logger.info("Executing login flow...")
        
        if self.login_flow.execute():
            logger.info("Login flow complete")
            self._transition_to(BotState.ENTER_END)
        else:
            logger.error("Login flow failed")
            self._transition_to(BotState.ERROR)
    
    def _handle_enter_end(self) -> None:
        """Handle ENTER_END state."""
        dimension = self.client.get_dimension()
        
        if dimension == "minecraft:the_end":
            logger.info("Already in The End")
            self._transition_to(BotState.SEARCH_FOR_CITY)
        else:
            logger.info("Traveling to The End...")
            # Execute go-to-end commands
            for cmd in self.config.go_to_end_commands:
                self.client.send_chat(cmd)
                time.sleep(self.config.command_delay)
            
            # Wait and check dimension
            time.sleep(3.0)
            if self.client.get_dimension() == "minecraft:the_end":
                self._transition_to(BotState.SEARCH_FOR_CITY)
            else:
                logger.warning("Not in The End yet, waiting...")
                time.sleep(5.0)
    
    def _handle_search_for_city(self) -> None:
        """Handle SEARCH_FOR_CITY state."""
        logger.info("Searching for End Cities...")
        
        # Get world snapshot
        world_data = self.perception.get_world_snapshot(
            radius=self.config.search_radius_blocks
        )
        
        # Find potential cities
        cities = self.scanner.find_potential_end_cities(world_data)
        
        # Filter out already searched cities
        new_cities = [c for c in cities if c not in self._searched_cities]
        
        if new_cities:
            # Select best candidate
            self._current_city = max(new_cities, key=lambda c: c.confidence)
            logger.info(f"Found city candidate at ({self._current_city.x}, "
                       f"{self._current_city.y}, {self._current_city.z}) "
                       f"confidence={self._current_city.confidence:.2f}")
            self._transition_to(BotState.PATH_TO_CITY)
        else:
            # No cities found, explore more
            logger.info("No cities found, exploring...")
            self._transition_to(BotState.MOVE_TO_NEXT_TARGET)
    
    def _handle_path_to_city(self) -> None:
        """Handle PATH_TO_CITY state."""
        if self._current_city is None:
            self._transition_to(BotState.SEARCH_FOR_CITY)
            return
        
        target = Position(
            self._current_city.x,
            self._current_city.y,
            self._current_city.z
        )
        
        if self.config.control_mode == "rl" and self._policy:
            # Use RL policy for navigation
            self._step_with_policy()
        else:
            # Use scripted navigation
            reached = self.navigator.walk_to(target)
            
            if reached:
                logger.info("Reached city")
                self._searched_cities.append(self._current_city)
                self._transition_to(BotState.SEARCH_FOR_SHIP)
    
    def _handle_search_for_ship(self) -> None:
        """Handle SEARCH_FOR_SHIP state."""
        logger.info("Searching for End Ship...")
        
        world_data = self.perception.get_world_snapshot(radius=64)
        ships = self.scanner.find_potential_ships(
            world_data,
            [self._current_city] if self._current_city else []
        )
        
        if ships:
            self._current_ship = max(ships, key=lambda s: s.confidence)
            logger.info(f"Found ship at ({self._current_ship.x}, "
                       f"{self._current_ship.y}, {self._current_ship.z})")
            self._transition_to(BotState.PATH_TO_SHIP)
        else:
            logger.info("No ship found at this city")
            # Log a "no elytra" result for this city
            self._record_find(elytra_found=False)
            self._transition_to(BotState.MOVE_TO_NEXT_TARGET)
    
    def _handle_path_to_ship(self) -> None:
        """Handle PATH_TO_SHIP state."""
        if self._current_ship is None:
            self._transition_to(BotState.SEARCH_FOR_SHIP)
            return
        
        target = Position(
            self._current_ship.x,
            self._current_ship.y,
            self._current_ship.z
        )
        
        reached = self.navigator.walk_to(target)
        
        if reached:
            logger.info("Reached ship")
            self._transition_to(BotState.OPEN_SHIP_CHEST)
    
    def _handle_open_ship_chest(self) -> None:
        """Handle OPEN_SHIP_CHEST state."""
        logger.info("Opening ship chest...")
        
        if self._current_ship is None:
            self._transition_to(BotState.ERROR)
            return
        
        # Ship chest is typically at a known relative position
        # For End Ships, the chest is inside the hull
        chest_pos = self.scanner.find_ship_chest_position(self._current_ship)
        
        if chest_pos:
            if self.client.open_container(chest_pos[0], chest_pos[1], chest_pos[2]):
                time.sleep(0.5)  # Wait for container to open
                self._transition_to(BotState.CHECK_FOR_ELYTRA)
            else:
                logger.warning("Failed to open chest")
                self._transition_to(BotState.ERROR)
        else:
            logger.warning("Could not find chest position")
            self._transition_to(BotState.MOVE_TO_NEXT_TARGET)
    
    def _handle_check_for_elytra(self) -> None:
        """Handle CHECK_FOR_ELYTRA state."""
        logger.info("Checking for Elytra...")
        
        items = self.client.get_container_items()
        
        if self.inventory.has_elytra(items):
            logger.info("🎉 ELYTRA FOUND!")
            self._record_find(elytra_found=True)
        else:
            logger.info("No Elytra in this chest")
            self._record_find(elytra_found=False)
        
        self.client.close_container()
        self._transition_to(BotState.LOG_RESULT)
    
    def _handle_log_result(self) -> None:
        """Handle LOG_RESULT state."""
        # Save finds to file
        self._save_finds()
        
        # Continue searching
        self._current_ship = None
        self._transition_to(BotState.MOVE_TO_NEXT_TARGET)
    
    def _handle_move_to_next(self) -> None:
        """Handle MOVE_TO_NEXT_TARGET state."""
        logger.info("Moving to next search area...")
        
        # Get a new exploration target
        current_pos = self.client.get_position()
        if current_pos is None:
            self._transition_to(BotState.ERROR)
            return
        
        # Move in a spiral pattern outward from origin
        import math
        search_index = len(self._searched_cities)
        angle = search_index * 0.5  # Radians
        distance = 500 + search_index * 200  # Blocks
        
        target_x = current_pos.x + distance * math.cos(angle)
        target_z = current_pos.z + distance * math.sin(angle)
        
        target = Position(target_x, 64, target_z)
        
        # Safe navigation (avoid void)
        self.navigator.safe_travel_to(target, max_void_risk=self.config.max_void_risk)
        
        self._current_city = None
        self._transition_to(BotState.SEARCH_FOR_CITY)
    
    def _handle_idle(self) -> None:
        """Handle IDLE state."""
        time.sleep(1.0)
    
    def _handle_error(self) -> None:
        """Handle ERROR state."""
        logger.error("Entered error state")
        
        # Try to recover
        time.sleep(5.0)
        
        if self.client.is_connected():
            # Try to continue from a safe state
            self._transition_to(BotState.SEARCH_FOR_CITY)
        else:
            # Try to reconnect
            self._transition_to(BotState.CONNECT)
    
    def _record_find(self, elytra_found: bool) -> None:
        """Record an Elytra find (or non-find)."""
        pos = self.client.get_position()
        if pos is None:
            return
        
        find = ElytraFind(
            timestamp=datetime.now().isoformat(),
            dimension=self.client.get_dimension(),
            x=pos.x,
            y=pos.y,
            z=pos.z,
            elytra_found=elytra_found,
            additional_info={
                "city_index": len(self._searched_cities),
                "ship_found": self._current_ship is not None
            }
        )
        
        self._elytra_finds.append(find)
        logger.info(f"Recorded find: {find.to_dict()}")
    
    def _save_finds(self) -> None:
        """Save Elytra finds to file."""
        try:
            with open(self.config.elytra_log_file, 'a') as f:
                for find in self._elytra_finds:
                    f.write(json.dumps(find.to_dict()) + "\n")
            logger.info(f"Saved {len(self._elytra_finds)} finds to "
                       f"{self.config.elytra_log_file}")
            self._elytra_finds.clear()
        except Exception as e:
            logger.error(f"Failed to save finds: {e}")
    
    def _check_runtime_limit(self) -> bool:
        """Check if runtime limit has been reached."""
        elapsed_minutes = (time.time() - self._start_time) / 60
        return elapsed_minutes >= self.config.max_runtime_minutes
    
    def _periodic_log(self) -> None:
        """Log periodic status updates."""
        now = time.time()
        if now - self._last_log_time >= self.config.log_interval_seconds:
            self._last_log_time = now
            
            pos = self.client.get_position()
            pos_str = f"({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})" if pos else "unknown"
            
            elapsed = (now - self._start_time) / 60
            
            logger.info(f"Status: state={self._state.name}, pos={pos_str}, "
                       f"cities_searched={len(self._searched_cities)}, "
                       f"runtime={elapsed:.1f}min")
    
    def _load_policy(self) -> None:
        """Load RL policy for inference."""
        try:
            from agent import Policy
            self._policy = Policy()
            logger.info("Loaded RL policy for inference")
        except Exception as e:
            logger.warning(f"Failed to load RL policy: {e}")
            logger.warning("Falling back to scripted control")
            self.config.control_mode = "scripted"
    
    def _step_with_policy(self) -> None:
        """Execute one step using the RL policy."""
        if self._policy is None:
            return
        
        # Get observation from perception
        obs = self.perception.get_observation_for_policy()
        
        # Get action from policy (inference mode)
        from agent.policy import select_action_from_observation
        action_idx = select_action_from_observation(self._policy.model, obs)
        
        # Convert action index to client commands
        self._execute_action(action_idx)
    
    def _execute_action(self, action_idx: int) -> None:
        """Execute an action index as client commands."""
        # Map action indices to movements
        action_map = {
            0: lambda: self.client.move(forward=1.0),
            1: lambda: self.client.move(forward=-1.0),
            2: lambda: self.client.move(strafe=-1.0),
            3: lambda: self.client.move(strafe=1.0),
            4: lambda: self.client.move(jump=True),
            5: lambda: self.client.look_at(*self._look_left()),
            6: lambda: self.client.look_at(*self._look_right()),
            7: lambda: self.client.use_item(),
        }
        
        action = action_map.get(action_idx, lambda: None)
        action()
    
    def _look_left(self) -> tuple:
        """Get position to look left."""
        pos = self.client.get_position()
        if pos:
            import math
            return (pos.x - 10, pos.y, pos.z)
        return (0, 64, 0)
    
    def _look_right(self) -> tuple:
        """Get position to look right."""
        pos = self.client.get_position()
        if pos:
            return (pos.x + 10, pos.y, pos.z)
        return (0, 64, 0)
    
    def _shutdown(self) -> None:
        """Clean shutdown."""
        logger.info("Shutting down...")
        
        # Save any remaining finds
        if self._elytra_finds:
            self._save_finds()
        
        # Disconnect
        self.client.disconnect()
        
        logger.info("Shutdown complete")
    
    def get_state(self) -> BotState:
        """Get current bot state."""
        return self._state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bot statistics."""
        elapsed = (time.time() - self._start_time) / 60
        
        return {
            "state": self._state.name,
            "runtime_minutes": elapsed,
            "cities_searched": len(self._searched_cities),
            "elytra_found": sum(1 for f in self._elytra_finds if f.elytra_found),
            "total_finds_logged": len(self._elytra_finds),
        }
