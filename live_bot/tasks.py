"""
tasks.py - Reusable task primitives for Elytra Finder Bot.

This module provides high-level task primitives:
- goto: Navigate to a position
- bridge: Bridge over a gap
- tower: Build up to a height
- descend: Safely descend
- wait_for_condition: Wait for a condition to be true
"""

import time
import logging
from typing import Callable, Optional, Any
from enum import Enum, auto

from integration.mc_client import MinecraftClient, Position
from .navigation import Navigator

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a task execution."""
    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    CANCELLED = auto()


class Task:
    """Base class for tasks."""
    
    def __init__(self, name: str):
        self.name = name
        self.status = TaskStatus.PENDING
        self.error: Optional[str] = None
    
    def execute(self) -> TaskStatus:
        """Execute the task. Override in subclasses."""
        raise NotImplementedError
    
    def cancel(self) -> None:
        """Cancel the task."""
        self.status = TaskStatus.CANCELLED


class GoToTask(Task):
    """Task to navigate to a position."""
    
    def __init__(
        self,
        navigator: Navigator,
        target: Position,
        tolerance: float = 2.0,
        timeout: float = 60.0
    ):
        super().__init__(f"GoTo({target.x:.0f}, {target.y:.0f}, {target.z:.0f})")
        self.navigator = navigator
        self.target = target
        self.tolerance = tolerance
        self.timeout = timeout
    
    def execute(self) -> TaskStatus:
        self.status = TaskStatus.RUNNING
        logger.info(f"Executing {self.name}")
        
        success = self.navigator.walk_to(
            self.target,
            timeout=self.timeout,
            tolerance=self.tolerance
        )
        
        if success:
            self.status = TaskStatus.SUCCESS
            logger.info(f"{self.name} succeeded")
        else:
            self.status = TaskStatus.FAILED
            self.error = "Failed to reach target"
            logger.warning(f"{self.name} failed")
        
        return self.status


class BridgeTask(Task):
    """Task to bridge over a gap."""
    
    def __init__(
        self,
        navigator: Navigator,
        target: Position,
        max_gap: int = 10
    ):
        super().__init__(f"Bridge to ({target.x:.0f}, {target.z:.0f})")
        self.navigator = navigator
        self.target = target
        self.max_gap = max_gap
    
    def execute(self) -> TaskStatus:
        self.status = TaskStatus.RUNNING
        logger.info(f"Executing {self.name}")
        
        success = self.navigator.bridge_over_gap(
            self.target,
            max_gap=self.max_gap
        )
        
        if success:
            self.status = TaskStatus.SUCCESS
        else:
            self.status = TaskStatus.FAILED
            self.error = "Failed to bridge"
        
        return self.status


class TowerTask(Task):
    """Task to build up to a height."""
    
    def __init__(
        self,
        navigator: Navigator,
        target_y: int,
        max_height: int = 20
    ):
        super().__init__(f"Tower to Y={target_y}")
        self.navigator = navigator
        self.target_y = target_y
        self.max_height = max_height
    
    def execute(self) -> TaskStatus:
        self.status = TaskStatus.RUNNING
        logger.info(f"Executing {self.name}")
        
        success = self.navigator.tower_up(
            self.target_y,
            max_height=self.max_height
        )
        
        if success:
            self.status = TaskStatus.SUCCESS
        else:
            self.status = TaskStatus.FAILED
            self.error = "Failed to tower"
        
        return self.status


class DescendTask(Task):
    """Task to safely descend."""
    
    def __init__(
        self,
        navigator: Navigator,
        target_y: int,
        max_fall: int = 3
    ):
        super().__init__(f"Descend to Y={target_y}")
        self.navigator = navigator
        self.target_y = target_y
        self.max_fall = max_fall
    
    def execute(self) -> TaskStatus:
        self.status = TaskStatus.RUNNING
        logger.info(f"Executing {self.name}")
        
        success = self.navigator.descend_safely(
            self.target_y,
            max_fall=self.max_fall
        )
        
        if success:
            self.status = TaskStatus.SUCCESS
        else:
            self.status = TaskStatus.FAILED
            self.error = "Failed to descend safely"
        
        return self.status


class WaitTask(Task):
    """Task to wait for a condition."""
    
    def __init__(
        self,
        condition: Callable[[], bool],
        timeout: float = 30.0,
        poll_interval: float = 0.5,
        name: str = "Wait"
    ):
        super().__init__(name)
        self.condition = condition
        self.timeout = timeout
        self.poll_interval = poll_interval
    
    def execute(self) -> TaskStatus:
        self.status = TaskStatus.RUNNING
        logger.info(f"Executing {self.name}")
        
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            if self.status == TaskStatus.CANCELLED:
                return self.status
            
            try:
                if self.condition():
                    self.status = TaskStatus.SUCCESS
                    return self.status
            except Exception as e:
                logger.warning(f"Condition check error: {e}")
            
            time.sleep(self.poll_interval)
        
        self.status = TaskStatus.FAILED
        self.error = "Timeout waiting for condition"
        return self.status


class SendChatTask(Task):
    """Task to send a chat message."""
    
    def __init__(
        self,
        client: MinecraftClient,
        message: str
    ):
        super().__init__(f"Chat: {message[:30]}...")
        self.client = client
        self.message = message
    
    def execute(self) -> TaskStatus:
        self.status = TaskStatus.RUNNING
        logger.info(f"Executing {self.name}")
        
        if self.client.send_chat(self.message):
            self.status = TaskStatus.SUCCESS
        else:
            self.status = TaskStatus.FAILED
            self.error = "Failed to send chat"
        
        return self.status


class TaskExecutor:
    """
    Executor for running tasks sequentially or in parallel.
    
    Usage:
        executor = TaskExecutor()
        executor.add_task(GoToTask(nav, target))
        executor.add_task(WaitTask(lambda: True))
        executor.run_all()
    """
    
    def __init__(self):
        self.tasks: list[Task] = []
        self.current_task: Optional[Task] = None
    
    def add_task(self, task: Task) -> 'TaskExecutor':
        """Add a task to the queue."""
        self.tasks.append(task)
        return self
    
    def clear(self) -> None:
        """Clear all pending tasks."""
        self.tasks.clear()
        self.current_task = None
    
    def run_all(self) -> bool:
        """
        Run all tasks sequentially.
        
        Returns:
            True if all tasks succeeded
        """
        for task in self.tasks:
            self.current_task = task
            status = task.execute()
            
            if status == TaskStatus.CANCELLED:
                logger.info("Task execution cancelled")
                return False
            
            if status == TaskStatus.FAILED:
                logger.warning(f"Task failed: {task.name} - {task.error}")
                return False
        
        self.current_task = None
        return True
    
    def cancel_current(self) -> None:
        """Cancel the currently running task."""
        if self.current_task:
            self.current_task.cancel()
    
    def get_status(self) -> dict:
        """Get status of task execution."""
        return {
            "pending_tasks": len(self.tasks),
            "current_task": self.current_task.name if self.current_task else None,
            "current_status": self.current_task.status.name if self.current_task else None
        }


# Convenience functions for creating tasks

def goto(
    navigator: Navigator,
    x: float,
    y: float,
    z: float,
    tolerance: float = 2.0
) -> GoToTask:
    """Create a goto task."""
    return GoToTask(navigator, Position(x, y, z), tolerance=tolerance)


def bridge_to(
    navigator: Navigator,
    x: float,
    y: float,
    z: float
) -> BridgeTask:
    """Create a bridge task."""
    return BridgeTask(navigator, Position(x, y, z))


def tower_to(
    navigator: Navigator,
    y: int
) -> TowerTask:
    """Create a tower task."""
    return TowerTask(navigator, y)


def wait_for(
    condition: Callable[[], bool],
    timeout: float = 30.0,
    name: str = "Wait"
) -> WaitTask:
    """Create a wait task."""
    return WaitTask(condition, timeout=timeout, name=name)


def wait_seconds(seconds: float) -> WaitTask:
    """Create a task that waits for a fixed time."""
    start = time.time()
    return WaitTask(
        lambda: time.time() - start >= seconds,
        timeout=seconds + 1,
        name=f"Wait {seconds}s"
    )


def send_chat(client: MinecraftClient, message: str) -> SendChatTask:
    """Create a send chat task."""
    return SendChatTask(client, message)
