# DOT
DOT is an intelligent multi-agent system that seamlessly connects documents with answers. Just like connecting dots to reveal a complete picture, DOT orchestrates specialized agents to parse your documents, understand your questions, and deliver precise responses with source attribution.
### Multi-Agent System
- **IngestionAgent**: Parses and preprocesses documents (PDF, DOCX, PPTX, CSV, TXT, MD)
- **RetrievalAgent**: Handles embedding generation and semantic retrieval using FAISS
- **LLMResponseAgent**: Generates responses using OpenAI GPT with retrieved context
- **CoordinatorAgent**: Orchestrates communication between agents

### Model Context Protocol (MCP)
All agents communicate using structured MCP messages:
```json
{
  "sender": "RetrievalAgent",
  "receiver": "LLMResponseAgent", 
  "type": "RETRIEVAL_RESPONSE",
  "trace_id": "abc-123",
  "payload": {
    "retrieved_chunks": ["...", "..."],
    "query": "What are the KPIs?"
  }
}
