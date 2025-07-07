from dataclasses import dataclass
from typing import Any, Dict, Optional
import uuid
from enum import Enum
import json

class MessageType(Enum):
    INGESTION_REQUEST = "INGESTION_REQUEST"
    INGESTION_RESPONSE = "INGESTION_RESPONSE"
    RETRIEVAL_REQUEST = "RETRIEVAL_REQUEST"
    RETRIEVAL_RESPONSE = "RETRIEVAL_RESPONSE"
    LLM_REQUEST = "LLM_REQUEST"
    LLM_RESPONSE = "LLM_RESPONSE"
    CONTEXT_RESPONSE = "CONTEXT_RESPONSE"
    ERROR = "ERROR"

@dataclass
class MCPMessage:
    sender: str
    receiver: str
    type: MessageType
    trace_id: str
    payload: Dict[str, Any]
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.type.value,
            "trace_id": self.trace_id,
            "payload": self.payload,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPMessage':
        return cls(
            sender=data["sender"],
            receiver=data["receiver"],
            type=MessageType(data["type"]),
            trace_id=data["trace_id"],
            payload=data["payload"],
            timestamp=data.get("timestamp")
        )

def generate_trace_id() -> str:
    return str(uuid.uuid4())[:8]