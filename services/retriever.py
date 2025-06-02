from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from typing import List
import os

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "pdf-data"
SIMILARITY_THRESHOLD = 0.4

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

class PineconeRetrieverWithThreshold(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> List[Document]:
        index = pc.Index(INDEX_NAME)
        vector = embeddings.embed_query(query)

        results = index.query(
            vector=vector,
            top_k=3,
            include_metadata=True
        )

        documents = []
        for match in results.matches:
            if match.score >= SIMILARITY_THRESHOLD:
                metadata = match.metadata or {}
                documents.append(Document(
                    page_content=metadata.get("text", ""),  # Match your stored key
                    metadata={
                        "page": metadata.get("page", ""),
                        "source": metadata.get("source", "")
                    }
                ))
        return documents

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)
