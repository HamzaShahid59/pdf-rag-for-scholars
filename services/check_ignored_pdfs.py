import os
from dotenv import load_dotenv
from pinecone import Pinecone
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PDF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pdfs"))
INDEX_NAME = "pdf-data"

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def check_pdf_sources():
    try:
        index = pc.Index(INDEX_NAME)
        missing_sources = []

        # Get all PDF filenames (without extension)
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
        logger.info(f"Found {len(pdf_files)} PDFs in folder.")

        for pdf_file in pdf_files:
            source_name = pdf_file.replace(".pdf", "")
            logger.info(f"Checking source: {source_name}")

            # Query to check if vectors exist with this source
            response = index.query(
                vector=[0.0] * 1536,
                filter={"source": {"$eq": pdf_file}},
                top_k=1,
                include_metadata=False
            )

            matches = response.get("matches", [])
            if not matches:
                logger.info(f"❌ No vectors found for: {source_name}")
                missing_sources.append(pdf_file)
            else:
                logger.info(f"✅ Vectors exist for: {source_name}")

        # Print final missing list
        print("\nLength of Found PDFS are " , len(missing_sources) )
        print("\n==== PDFs with NO vectors in Pinecone ====")
        for missing in missing_sources:
            print(f"• {missing}")

    except Exception as e:
        logger.error(f"Error checking sources: {e}")

if __name__ == "__main__":
    check_pdf_sources()
