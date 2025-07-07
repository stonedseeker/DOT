import asyncio
from typing import Dict, Any, List
from .base_agent import BaseAgent
from mcp.message_protocol import MCPMessage, MessageType
from utils.vector_store import VectorStore
from utils.embeddings import EmbeddingGenerator
import os

class RetrievalAgent(BaseAgent):
    def __init__(self):
        super().__init__("RetrievalAgent")
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore(self.embedding_generator.get_embedding_dimension())
        self.documents_indexed = set()
    
    async def handle_message(self, message: MCPMessage):
        """Handle incoming messages"""
        if message.type == MessageType.INGESTION_RESPONSE:
            await self._process_ingestion_response(message)
        elif message.type == MessageType.RETRIEVAL_REQUEST:
            await self._process_retrieval_request(message)
    
    async def _process_ingestion_response(self, message: MCPMessage):
        """Process ingested documents and add to vector store"""
        try:
            document_id = message.payload.get('document_id')
            text_chunks = message.payload.get('text_chunks', [])
            metadata = message.payload.get('metadata', {})
            
            if document_id in self.documents_indexed:
                self.log_info(f"Document {document_id} already indexed")
                return
            
            self.log_info(f"Indexing document: {document_id}")
            
            # Extract texts for embedding
            texts = [chunk['text'] for chunk in text_chunks]
            
            if not texts:
                self.log_info(f"No text content found in document: {document_id}")
                return
            
            # Generate embeddings
            embeddings = self.embedding_generator.generate_embeddings(texts)
            
            # Prepare metadata for each chunk
            chunk_metadata = []
            for i, chunk in enumerate(text_chunks):
                chunk_meta = {
                    'document_id': document_id,
                    'chunk_id': i,
                    'document_metadata': metadata,
                    'chunk_metadata': chunk.get('metadata', {})
                }
                chunk_metadata.append(chunk_meta)
            
            # Add to vector store
            self.vector_store.add_documents(embeddings, texts, chunk_metadata)
            self.documents_indexed.add(document_id)
            
            self.log_info(f"Successfully indexed {len(texts)} chunks for document: {document_id}")
            
        except Exception as e:
            self.log_error(f"Error indexing document: {e}")
    
    async def _process_retrieval_request(self, message: MCPMessage):
        """Process retrieval request"""
        try:
            query = message.payload.get('query')
            top_k = message.payload.get('top_k', 5)
            
            self.log_info(f"Processing retrieval request: {query}")
            
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embeddings([query])[0]
            
            # Search in vector store
            search_results = self.vector_store.search(query_embedding, k=top_k)
            
            # Format results
            retrieved_chunks = []
            for result in search_results:
                retrieved_chunks.append({
                    'text': result['document'],
                    'metadata': result['metadata'],
                    'score': result['score']
                })
            
            # Send response to LLMResponseAgent
            await self.send_message(
                receiver="LLMResponseAgent",
                message_type=MessageType.RETRIEVAL_RESPONSE,
                payload={
                    'query': query,
                    'retrieved_chunks': retrieved_chunks,
                    'total_results': len(retrieved_chunks)
                },
                trace_id=message.trace_id
            )
            
            self.log_info(f"Retrieved {len(retrieved_chunks)} chunks for query")
            
        except Exception as e:
            self.log_error(f"Error processing retrieval request: {e}")
            await self.send_message(
                receiver=message.sender,
                message_type=MessageType.ERROR,
                payload={'error': str(e)},
                trace_id=message.trace_id
            )