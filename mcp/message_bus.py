import asyncio
from typing import Dict, List, Callable
from .message_protocol import MCPMessage
import logging

class MessageBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history: List[MCPMessage] = []
        self.logger = logging.getLogger(__name__)
    
    def subscribe(self, agent_name: str, callback: Callable):
        """Subscribe an agent to receive messages"""
        if agent_name not in self.subscribers:
            self.subscribers[agent_name] = []
        self.subscribers[agent_name].append(callback)
    
    async def publish(self, message: MCPMessage):
        """Publish a message to the intended receiver"""
        self.message_history.append(message)
        self.logger.info(f"Publishing message: {message.sender} -> {message.receiver} ({message.type.value})")
        
        if message.receiver in self.subscribers:
            for callback in self.subscribers[message.receiver]:
                try:
                    await callback(message)
                except Exception as e:
                    self.logger.error(f"Error in callback for {message.receiver}: {e}")
    
    def get_message_history(self, trace_id: str = None) -> List[MCPMessage]:
        """Get message history, optionally filtered by trace_id"""
        if trace_id:
            return [msg for msg in self.message_history if msg.trace_id == trace_id]
        return self.message_history

# Global message bus instance
message_bus = MessageBus()