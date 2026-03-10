"""Base agent class for all specialized agents"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from loguru import logger


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    Follows OOP principles with clear interface definition.
    """
    
    def __init__(self, name: str):
        """
        Initialize base agent.
        
        Args:
            name: Name of the agent
        """
        self.name = name
        logger.info(f"Initializing {self.name} agent")
    
    @abstractmethod
    def process_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Process a query and return results.
        
        Args:
            query: Natural language query
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing results and metadata
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get agent capabilities and metadata.
        
        Returns:
            Dictionary describing agent capabilities
        """
        pass
    
    def validate_query(self, query: str) -> bool:
        """
        Validate if query is appropriate for this agent.
        
        Args:
            query: Query string
            
        Returns:
            True if valid, False otherwise
        """
        if not query or not isinstance(query, str):
            return False
        return len(query.strip()) > 0
    
    def format_response(
        self,
        success: bool,
        data: Any = None,
        error: str = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Format agent response in a standard structure.
        
        Args:
            success: Whether the operation was successful
            data: Result data
            error: Error message if any
            metadata: Additional metadata
            
        Returns:
            Formatted response dictionary
        """
        response = {
            'agent': self.name,
            'success': success,
            'data': data,
            'error': error,
            'metadata': metadata or {}
        }
        return response
