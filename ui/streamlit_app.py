import streamlit as st
import asyncio
import os
import tempfile
from typing import Dict, Any
import sys
# sys.path.append('..')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from agents.coordinator_agent import CoordinatorAgent
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import LLMResponseAgent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize agents
@st.cache_resource
def initialize_agents():
    """Initialize all agents"""
    ingestion_agent = IngestionAgent()
    retrieval_agent = RetrievalAgent()
    llm_response_agent = LLMResponseAgent()
    coordinator_agent = CoordinatorAgent()
    
    return {
        'ingestion': ingestion_agent,
        'retrieval': retrieval_agent,
        'llm': llm_response_agent,
        'coordinator': coordinator_agent
    }

def main():
    st.set_page_config(
        page_title="Agentic RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("Agentic RAG Chatbot with MCP")
    st.markdown("Upload documents and ask questions using our multi-agent system!")
    
    # Initialize agents
    agents = initialize_agents()
    coordinator = agents['coordinator']
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'docx', 'pptx', 'csv', 'txt', 'md'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in [f['name'] for f in st.session_state.uploaded_files]:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Process document
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        result = asyncio.run(coordinator.process_document_upload(
                            tmp_path, 
                            uploaded_file.name.split('.')[-1]
                        ))
                    
                    st.session_state.uploaded_files.append({
                        'name': uploaded_file.name,
                        'path': tmp_path,
                        'status': 'processed'
                    })
                    
                    st.success(f"✅ {uploaded_file.name} processed successfully!")
        
        # Display uploaded files
        if st.session_state.uploaded_files:
            st.subheader("📄 Uploaded Files")
            for file_info in st.session_state.uploaded_files:
                st.write(f"• {file_info['name']}")
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.write(f"**Source {i+1}:** {source['document_id']} (Section: {source['section']}, Score: {source['score']:.3f})")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from coordinator
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = asyncio.run(coordinator.process_user_query(prompt))
            
            st.markdown(response.get('response', 'No response generated'))
            
            # Add assistant message to chat
            assistant_message = {
                "role": "assistant", 
                "content": response.get('response', 'No response generated')
            }
            
            if 'sources' in response:
                assistant_message["sources"] = response['sources']
                
                with st.expander("📚 Sources"):
                    for i, source in enumerate(response['sources']):
                        st.write(f"**Source {i+1}:** {source['document_id']} (Section: {source['section']}, Score: {source['score']:.3f})")
            
            st.session_state.messages.append(assistant_message)
    
    # System information
    with st.expander("🔧 System Information"):
        st.write("**Agents Status:**")
        st.write("Ingestion Agent: Active")
        st.write("Retrieval Agent: Active") 
        st.write("LLM Response Agent: Active")
        st.write("Coordinator Agent: Active")
        
        st.write(f"**Documents Processed:** {len(st.session_state.uploaded_files)}")

if __name__ == "__main__":
    main()