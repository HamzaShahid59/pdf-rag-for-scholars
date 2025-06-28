import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
import logging
import openai
from openai import OpenAI
from openai import OpenAIError
import time
client = OpenAI()

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "pdf-data"
# INDEX_NAME = "test-pdf-data-1"
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



def get_embedding_with_retries(text, retries=3, delay=5):
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


def create_embeddings(documents):
    try:
        initialize_pinecone_index()
        index = pc.Index(INDEX_NAME)
        count = 0

        for doc in documents:
            print(f"[Page {doc.metadata.get('page', '?')}] from {doc.metadata.get('filename', '')}")

            details = doc.page_content.strip()
            if not details:
                continue

            unique_id = str(hash(details))
            embedding = get_embedding_with_retries(details)

            vector = (
                unique_id,
                embedding,
                {
                    "text": details,
                    "source": doc.metadata.get("source", doc.metadata.get("filename", "")),
                    "page": doc.metadata.get("page", ""),
                }
            )

            # ✅ Store in the specified namespace
            index.upsert([vector], namespace="altas-copco-sf-manuals")
            count += 1

        print(f"✅ Successfully stored {count} embeddings in Pinecone namespace 'altas-copco-sf-manuals'.")

    except Exception as e:
        logger.error(f"Embedding creation failed: {e}")
