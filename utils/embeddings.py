import openai
import numpy as np
from typing import List
import logging
import os
import streamlit as st  # <--- Important for Streamlit Cloud

class EmbeddingGenerator:
    def __init__(self, model_name: str = "text-embedding-3-large", api_key: str = None):
        self.model_name = model_name
        self.api_key = (
            api_key
            or st.secrets.get("OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )

        self.logger = logging.getLogger(__name__)
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as environment variable 'OPENAI_API_KEY'")

        openai.api_key = self.api_key

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        try:
            response = openai.embeddings.create(
                model=self.model_name,
                input=texts
            )
            embeddings = [d.embedding for d in response.data]
            return np.array(embeddings)
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        return 3072
