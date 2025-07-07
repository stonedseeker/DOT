import asyncio
from typing import Dict, Any
from .base_agent import BaseAgent
from mcp.message_protocol import MCPMessage, MessageType, generate_trace_id
import os

class CoordinatorAgent(BaseAgent):
    def __init__(self):
        super().__init__("CoordinatorAgent")
        self.active_conversations = {}
    
    async def handle_message(self, message: MCPMessage):
        """Handle incoming messages"""
        if message.type == MessageType.LLM_RESPONSE:
            await self._process_llm_response(message)
        elif message.type == MessageType.ERROR:
            await self._handle_error(message)
    
    async def process_user_query(self, query: str, conversation_id: str = None) -> Dict[str, Any]:
        """Process user query and coordinate between agents"""
        trace_id = generate_trace_id()
        
        if conversation_id:
            if conversation_id not in self.active_conversations:
                self.active_conversations[conversation_id] = []
            self.active_conversations[conversation_id].append({
                'query': query,
                'trace_id': trace_id
            })
        
        self.log_info(f"Processing user query: {query}")
        
        # Create a future to wait for the response
        response_future = asyncio.Future()
        self.active_conversations[trace_id] = response_future
        
        # Send retrieval request
        await self.send_message(
            receiver="RetrievalAgent",
            message_type=MessageType.RETRIEVAL_REQUEST,
            payload={
                'query': query,
                'top_k': 5
            },
            trace_id=trace_id
        )
        
        # Wait for response
        try:
            response = await asyncio.wait_for(response_future, timeout=30.0)
            return response
        except asyncio.TimeoutError:
            self.log_error(f"Timeout waiting for response to query: {query}")
            return {
                'query': query,
                'response': "I apologize, but the request timed out. Please try again.",
                'sources': [],
                'error': 'timeout'
            }
    
    async def process_document_upload(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """Process document upload"""
        trace_id = generate_trace_id()
        
        self.log_info(f"Processing document upload: {file_path}")
        
        # Send ingestion request
        await self.send_message(
            receiver="IngestionAgent",
            message_type=MessageType.INGESTION_REQUEST,
            payload={
                'file_path': file_path,
                'file_type': file_type
            },
            trace_id=trace_id
        )
        
        return {
            'status': 'processing',
            'file_path': file_path,
            'file_type': file_type,
            'trace_id': trace_id
        }
    
    async def _process_llm_response(self, message: MCPMessage):
        """Process LLM response"""
        trace_id = message.trace_id
        
        if trace_id in self.active_conversations:
            future = self.active_conversations[trace_id]
            if not future.done():
                future.set_result(message.payload)
            del self.active_conversations[trace_id]
    
    async def _handle_error(self, message: MCPMessage):
        """Handle error messages"""
        trace_id = message.trace_id
        
        if trace_id in self.active_conversations:
            future = self.active_conversations[trace_id]
            if not future.done():
                future.set_result({
                    'error': message.payload.get('error', 'Unknown error'),
                    'response': 'An error occurred while processing your request.'
                })
            del self.active_conversations[trace_id]