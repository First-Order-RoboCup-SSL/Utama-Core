from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")
class AbstractPID(ABC, Generic[T]):
    
    @abstractmethod
    def calculate(self, target: T, current: T) -> T:
        """Perform a PID calculation."""
        ...
    
    @abstractmethod
    def reset(self, robot_id: int):
        """Reset the PID controller state for a given robot."""
        ...