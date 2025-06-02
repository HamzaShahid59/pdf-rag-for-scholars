import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
import logging
import openai

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "pdf-data"
SIMILARITY_THRESHOLD = 0.40
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def initialize_pinecone_index():
    existing_indexes = pc.list_indexes().names()
    if INDEX_NAME not in existing_indexes:
        logger.info("Creating Pinecone index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        logger.info("Index created successfully!")
    else:
        logger.info("Index already exists!")


def check_index_data():
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    return stats['total_vector_count'] > 0


def create_embeddings(documents):
    try:
        initialize_pinecone_index()
        # if check_index_data():
        #     logger.info("Index already has data. Skipping embedding creation.")
        #     return
        
        vectors = []
        count = 0
        index = pc.Index(INDEX_NAME)

        for doc in documents:
            details = doc.page_content.strip()
            if not details:
                continue

            unique_id = str(hash(details))  # Generate a unique ID

            # Generate embedding
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=details
            )
            embedding = response.data[0].embedding

            # Add to batch
            vectors.append((
                unique_id,
                embedding,
                {
                    "text": details,
                    "source": doc.metadata.get("source", ""),
                    "page": doc.metadata.get("page", ""),
                }
            ))
            count += 1

            # Batch upsert every 100 vectors
            if len(vectors) >= 100:
                index.upsert(vectors)
                vectors = []

        # Final batch
        if vectors:
            index.upsert(vectors)

        print(f"Successfully stored {count} embeddings in Pinecone.")

        
    except Exception as e:
        logger.error(f"Embedding creation failed: {e}")