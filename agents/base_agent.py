import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any
from mcp.message_protocol import MCPMessage, MessageType
from mcp.message_bus import message_bus
import logging

class BaseAgent(ABC):
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"{__name__}.{agent_name}")
        message_bus.subscribe(agent_name, self.handle_message)
    
    @abstractmethod
    async def handle_message(self, message: MCPMessage):
        """Handle incoming messages"""
        pass
    
    async def send_message(self, receiver: str, message_type: MessageType, 
                          payload: Dict[str, Any], trace_id: str):
        """Send a message to another agent"""
        message = MCPMessage(
            sender=self.agent_name,
            receiver=receiver,
            type=message_type,
            trace_id=trace_id,
            payload=payload
        )
        await message_bus.publish(message)
    
    def log_info(self, message: str):
        self.logger.info(f"[{self.agent_name}] {message}")
    
    def log_error(self, message: str):
        self.logger.error(f"[{self.agent_name}] {message}")