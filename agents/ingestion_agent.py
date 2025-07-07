import asyncio
from typing import Dict, Any, List
from .base_agent import BaseAgent
from mcp.message_protocol import MCPMessage, MessageType
from utils.document_parsers import DocumentParser
import os

class IngestionAgent(BaseAgent):
    def __init__(self):
        super().__init__("IngestionAgent")
        self.parser = DocumentParser()
        self.processed_documents = {}
    
    async def handle_message(self, message: MCPMessage):
        """Handle incoming messages"""
        if message.type == MessageType.INGESTION_REQUEST:
            await self._process_ingestion_request(message)
    
    async def _process_ingestion_request(self, message: MCPMessage):
        """Process document ingestion request"""
        try:
            file_path = message.payload.get('file_path')
            file_type = message.payload.get('file_type')
            
            self.log_info(f"Processing {file_type} file: {file_path}")
            
            # Parse the document
            parsed_doc = self.parser.parse_document(file_path, file_type)
            
            # Extract text chunks for embedding
            text_chunks = self._extract_text_chunks(parsed_doc)
            
            # Store processed document
            doc_id = f"{file_path}_{file_type}"
            self.processed_documents[doc_id] = {
                'parsed_doc': parsed_doc,
                'text_chunks': text_chunks
            }
            
            # Send response to RetrievalAgent
            await self.send_message(
                receiver="RetrievalAgent",
                message_type=MessageType.INGESTION_RESPONSE,
                payload={
                    'document_id': doc_id,
                    'text_chunks': text_chunks,
                    'metadata': parsed_doc.get('metadata', {}),
                    'document_type': file_type
                },
                trace_id=message.trace_id
            )
            
            self.log_info(f"Successfully processed document: {doc_id}")
            
        except Exception as e:
            self.log_error(f"Error processing document: {e}")
            await self.send_message(
                receiver=message.sender,
                message_type=MessageType.ERROR,
                payload={'error': str(e)},
                trace_id=message.trace_id
            )
    
    def _extract_text_chunks(self, parsed_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract text chunks from parsed document"""
        chunks = []
        content = parsed_doc.get('content', [])
        
        for item in content:
            if isinstance(item, dict) and 'content' in item:
                chunks.append({
                    'text': item['content'],
                    'metadata': {
                        'document_type': parsed_doc['type'],
                        'section': item.get('page', item.get('slide', item.get('paragraph', 1)))
                    }
                })
        
        return chunks