from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
# from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from typing import List
import os
from openai import OpenAI, OpenAIError
import time
import logging


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# INDEX_NAME = "test-pdf-data-1"
INDEX_NAME = "pdf-data"
SIMILARITY_THRESHOLD = 0.1

# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# Setup
client = OpenAI()
logger = logging.getLogger(__name__)

def get_embedding_with_retries(text, retries=3, delay=5):
    """Create embeddings with retry logic - keep your existing function"""
    for attempt in range(1, retries + 1):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                timeout=60
            )
            return response.data[0].embedding
        except OpenAIError as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            if attempt == retries:
                raise
            time.sleep(delay * attempt)

class PineconeRetrieverWithThreshold(BaseRetriever):
    def __init__(self, namespace: str = ""):
        super().__init__()
        self._namespace = namespace  # Use underscore to avoid LangChain validation issues

    @property
    def namespace(self):
        return self._namespace

    def _get_relevant_documents(self, query: str) -> List[Document]:
        index = pc.Index(INDEX_NAME)
        vector = get_embedding_with_retries(query)
        print("Name space using is ")
        print(self.namespace)
        results = index.query(
            vector=vector,
            top_k=5,
            include_metadata=True,
            namespace=self.namespace  # Now safe
        )

        documents = []
        for match in results.matches:
            if match.score >= SIMILARITY_THRESHOLD:
                metadata = match.metadata or {}
                documents.append(Document(
                    page_content=metadata.get("text", ""),
                    metadata={
                        "page": metadata.get("page", ""),
                        "source": metadata.get("source", ""),
                        "filename": metadata.get("filename", ""),
                        "score": match.score
                    }
                ))
        return documents



