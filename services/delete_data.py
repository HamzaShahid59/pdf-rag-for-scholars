import os
from dotenv import load_dotenv
from pinecone import Pinecone
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "pdf-data"

# List of sources you want to delete
TARGET_SOURCES = [
    "atlas-copco-elektronikon-1-manual.pdf"
]
# TARGET_SOURCES = [
#     "atlas-copco-ga-200-manual.pdf",
#     "atlas-copco-ga-250-manual-pdf.pdf",
#     "atlas-copco-ga-90.pdf",
#     "atlas-copco-ga7-service-manual.pdf",
#     "atlas-copco-ga-22-ff-manual.pdf",
#     "atlas-copco-ga5-manual.pdf",
#     "atlas-copco-xas-65-manual.pdf",
#     "atlas-copco-ga22-manual-download.pdf"
# ]

def delete_vectors_by_sources(sources):
    try:
        index = pc.Index(INDEX_NAME)
        all_ids_to_delete = []

        for source in sources:
            logger.info(f"Searching for vectors with source: '{source}'")
            ids_to_delete = []
            cursor = None

            while True:
                response = index.query(
                    vector=[0.0] * 1536,  # Dummy vector
                    filter={"source": {"$eq": source}},
                    top_k=100,
                    include_metadata=True
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
        index.delete(ids=all_ids_to_delete)
        logger.info("✅ Deletion completed successfully.")

    except Exception as e:
        logger.error(f"Error during deletion: {e}")

if __name__ == "__main__":
    delete_vectors_by_sources(TARGET_SOURCES)
