import asyncio
from typing import Dict, Any, List
from .base_agent import BaseAgent
from mcp.message_protocol import MCPMessage, MessageType
import openai
import os
import streamlit as st

class LLMResponseAgent(BaseAgent):
    def __init__(self):
        super().__init__("LLMResponseAgent")
        # Initialize OpenAI client (you can replace with any LLM)
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=api_key)
    
    async def handle_message(self, message: MCPMessage):
        """Handle incoming messages"""
        if message.type == MessageType.RETRIEVAL_RESPONSE:
            await self._process_retrieval_response(message)
    
    async def _process_retrieval_response(self, message: MCPMessage):
        """Process retrieval response and generate LLM response"""
        try:
            query = message.payload.get('query')
            retrieved_chunks = message.payload.get('retrieved_chunks', [])
            
            self.log_info(f"Generating response for query: {query}")
            
            # Build context from retrieved chunks
            context = self._build_context(retrieved_chunks)
            
            # Generate prompt
            prompt = self._build_prompt(query, context)
            
            # Call LLM
            response = self._call_llm(prompt)
            
            # Send response back to coordinator
            await self.send_message(
                receiver="CoordinatorAgent",
                message_type=MessageType.LLM_RESPONSE,
                payload={
                    'query': query,
                    'response': response,
                    'context_used': retrieved_chunks,
                    'sources': self._extract_sources(retrieved_chunks)
                },
                trace_id=message.trace_id
            )
            
            self.log_info("Successfully generated LLM response")
            
        except Exception as e:
            self.log_error(f"Error generating LLM response: {e}")
            await self.send_message(
                receiver="CoordinatorAgent",
                message_type=MessageType.ERROR,
                payload={'error': str(e)},
                trace_id=message.trace_id
            )
    
    def _build_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Build context from retrieved chunks"""
        context_parts = []
        
        for i, chunk in enumerate(retrieved_chunks):
            metadata = chunk.get('metadata', {})
            document_id = metadata.get('document_id', 'Unknown')
            section = metadata.get('chunk_metadata', {}).get('section', 'Unknown')
            
            context_parts.append(f"[Source {i+1} - {document_id}, Section {section}]:\n{chunk['text']}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for LLM"""
        prompt = f"""You are a helpful assistant that answers questions based on the provided context. 
Use only the information from the context to answer the question. If the context doesn't contain 
enough information to answer the question, say so clearly.

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above. Include references to the sources when relevant."""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM to generate response"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            self.log_error(f"Error calling LLM: {e}")
            # Fallback response
            return "I apologize, but I encountered an error while generating the response. Please try again."
    
    def _extract_sources(self, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from retrieved chunks"""
        sources = []
        
        for chunk in retrieved_chunks:
            metadata = chunk.get('metadata', {})
            sources.append({
                'document_id': metadata.get('document_id', 'Unknown'),
                'section': metadata.get('chunk_metadata', {}).get('section', 'Unknown'),
                'score': chunk.get('score', 0.0)
            })
        
        return sources