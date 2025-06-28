import os
from dotenv import load_dotenv
from pinecone import Pinecone
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "pdf-data"
NAMESPACE = "altas-copco-xas-manuals"  # 👈 Define your target namespace

# List of sources you want to delete
TARGET_SOURCES = [
    "atlas-copco-xas-125-manual.pdf"
]

def delete_vectors_by_sources(sources, namespace):
    try:
        index = pc.Index(INDEX_NAME)
        all_ids_to_delete = []

        for source in sources:
            logger.info(f"Searching for vectors with source: '{source}' in namespace: '{namespace}'")
            ids_to_delete = []

            while True:
                response = index.query(
                    vector=[0.0] * 1536,  # Dummy vector
                    filter={"source": {"$eq": source}},
                    top_k=100,
                    include_metadata=True,
                    namespace=namespace  # 👈 Search within the namespace
                )
                matches = response.get("matches", [])
                ids_to_delete.extend([match["id"] for match in matches])

                if len(matches) < 100:
                    break  # No more results for this source

            if ids_to_delete:
                logger.info(f"Found {len(ids_to_delete)} vectors for source '{source}'.")
                all_ids_to_delete.extend(ids_to_delete)
            else:
                logger.info(f"No vectors found for source '{source}'.")

        if not all_ids_to_delete:
            logger.info("No matching vectors found to delete.")
            return

        logger.info(f"Deleting total {len(all_ids_to_delete)} vectors from Pinecone...")
        index.delete(ids=all_ids_to_delete, namespace=namespace)  # 👈 Delete from the namespace
        logger.info("✅ Deletion completed successfully.")

    except Exception as e:
        logger.error(f"Error during deletion: {e}")

if __name__ == "__main__":
    delete_vectors_by_sources(TARGET_SOURCES, NAMESPACE)
