"""
login_flow.py - Handle connection and server navigation for LemonCloud.

This module implements the scripted login flow:
1. Connect to LemonCloud IP
2. Execute hub/server commands
3. Navigate to survival/The End
4. Handle chat confirmations and delays
"""

import time
import logging
import re
from typing import List, Optional, Callable

from integration.mc_client import MinecraftClient

logger = logging.getLogger(__name__)


class LoginFlow:
    """
    Handles the LemonCloud server login and navigation flow.
    
    This class executes a configurable sequence of commands
    to get from initial connection to the target game world.
    
    The flow is resilient:
    - Waits for chat confirmations
    - Has configurable delays between commands
    - Handles common server messages
    
    Usage:
        flow = LoginFlow(client, ["/survival"], ["/warp end"])
        if flow.execute():
            print("Ready to search for Elytra!")
    """
    
    # Common server messages to detect
    SERVER_READY_PATTERNS = [
        r"teleport",
        r"warped",
        r"welcome",
        r"joined",
        r"connected",
        r"survival",
    ]
    
    ERROR_PATTERNS = [
        r"error",
        r"fail",
        r"invalid",
        r"unknown command",
        r"permission",
    ]
    
    def __init__(
        self,
        client: MinecraftClient,
        login_commands: List[str],
        go_to_end_commands: List[str],
        command_delay: float = 2.0,
        message_timeout: float = 10.0
    ):
        """
        Initialize the login flow.
        
        Args:
            client: Minecraft client for commands
            login_commands: Commands to join survival server
            go_to_end_commands: Commands to get to The End
            command_delay: Delay between commands
            message_timeout: Time to wait for confirmations
        """
        self.client = client
        self.login_commands = login_commands
        self.go_to_end_commands = go_to_end_commands
        self.command_delay = command_delay
        self.message_timeout = message_timeout
    
    def execute(self) -> bool:
        """
        Execute the complete login flow.
        
        Returns:
            True if successfully reached target state
        """
        logger.info("Starting login flow...")
        
        # Execute login commands
        if not self._execute_command_sequence(
            self.login_commands,
            "login"
        ):
            return False
        
        # Wait for server to be ready
        time.sleep(self.command_delay * 2)
        
        logger.info("Login flow complete")
        return True
    
    def execute_go_to_end(self) -> bool:
        """
        Execute commands to go to The End.
        
        Returns:
            True if successful
        """
        logger.info("Executing go-to-end flow...")
        
        if not self._execute_command_sequence(
            self.go_to_end_commands,
            "go_to_end"
        ):
            return False
        
        # Verify we're in The End
        time.sleep(self.command_delay)
        dimension = self.client.get_dimension()
        
        if dimension == "minecraft:the_end":
            logger.info("Successfully reached The End")
            return True
        else:
            logger.warning(f"Not in The End, current dimension: {dimension}")
            return False
    
    def _execute_command_sequence(
        self,
        commands: List[str],
        sequence_name: str
    ) -> bool:
        """
        Execute a sequence of commands.
        
        Args:
            commands: List of commands to execute
            sequence_name: Name for logging
            
        Returns:
            True if all commands executed successfully
        """
        logger.info(f"Executing {sequence_name} sequence: {commands}")
        
        for i, command in enumerate(commands):
            logger.info(f"[{i+1}/{len(commands)}] Sending: {command}")
            
            # Send command
            if not self.client.send_chat(command):
                logger.error(f"Failed to send command: {command}")
                return False
            
            # Wait for response
            time.sleep(self.command_delay)
            
            # Check for confirmation (optional)
            confirmation = self._wait_for_confirmation()
            if confirmation:
                logger.debug(f"Got confirmation: {confirmation}")
        
        return True
    
    def _wait_for_confirmation(
        self,
        patterns: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Wait for a chat message confirming command execution.
        
        Args:
            patterns: Regex patterns to match
            
        Returns:
            Matching message or None
        """
        if patterns is None:
            patterns = self.SERVER_READY_PATTERNS
        
        # This would use the client's chat message listening
        # For now, just wait
        start = time.time()
        
        while time.time() - start < self.message_timeout:
            # Check for messages from server
            # TODO: Implement actual chat message handling
            time.sleep(0.1)
        
        return None
    
    def wait_for_server_ready(
        self,
        timeout: float = 30.0
    ) -> bool:
        """
        Wait for server to be ready after joining.
        
        Args:
            timeout: Maximum time to wait
            
        Returns:
            True if server is ready
        """
        logger.info("Waiting for server to be ready...")
        
        start = time.time()
        
        while time.time() - start < timeout:
            # Check if connected and can move
            if self.client.is_connected():
                pos = self.client.get_position()
                if pos is not None:
                    logger.info("Server is ready")
                    return True
            
            time.sleep(0.5)
        
        logger.warning("Timeout waiting for server to be ready")
        return False
    
    def handle_queue(self, max_wait: float = 300.0) -> bool:
        """
        Handle server queue if present.
        
        Some servers put you in a queue before joining.
        
        Args:
            max_wait: Maximum time to wait in queue
            
        Returns:
            True if passed queue, False if timeout
        """
        logger.info("Checking for server queue...")
        
        start = time.time()
        
        while time.time() - start < max_wait:
            # Check for queue-related messages
            # TODO: Implement queue detection
            
            # Check if we're in the game
            if self.client.is_connected():
                dimension = self.client.get_dimension()
                if dimension and dimension != "":
                    logger.info("Passed queue, in game")
                    return True
            
            time.sleep(1.0)
        
        logger.warning("Queue wait timeout")
        return False


class CommandBuilder:
    """
    Helper for building command sequences.
    
    Usage:
        commands = CommandBuilder()
            .add("/server survival")
            .wait(2)
            .add("/warp end")
            .build()
    """
    
    def __init__(self):
        self._commands: List[dict] = []
    
    def add(self, command: str) -> 'CommandBuilder':
        """Add a command to the sequence."""
        self._commands.append({
            "type": "command",
            "value": command
        })
        return self
    
    def wait(self, seconds: float) -> 'CommandBuilder':
        """Add a wait to the sequence."""
        self._commands.append({
            "type": "wait",
            "value": seconds
        })
        return self
    
    def wait_for_message(
        self,
        pattern: str,
        timeout: float = 10.0
    ) -> 'CommandBuilder':
        """Add a wait-for-message to the sequence."""
        self._commands.append({
            "type": "wait_message",
            "pattern": pattern,
            "timeout": timeout
        })
        return self
    
    def build(self) -> List[dict]:
        """Build the command sequence."""
        return self._commands
    
    def get_commands_only(self) -> List[str]:
        """Get just the command strings."""
        return [
            c["value"] for c in self._commands
            if c["type"] == "command"
        ]
