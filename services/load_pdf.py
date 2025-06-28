import os
import fitz
import logging
import pytesseract
import tempfile
import re
from PIL import Image, ImageEnhance, ImageFilter
from typing import List
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, Image as ImageElement
from pdf2image import convert_from_path
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OpenAIEmbeddings

# Logger setup
logger = logging.getLogger("services.load_pdf")

# Paths
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r'C:\Users\abc\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin'
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def is_garbage(text: str) -> bool:
    nonprintable = sum(1 for c in text if not c.isprintable())
    ratio = nonprintable / max(len(text), 1)
    return ratio > 0.4

def is_binary_or_junk(text: str) -> bool:
    return sum(1 for c in text if ord(c) < 32 and c not in "\n\t\r") / max(len(text), 1) > 0.3

def clean_intro(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(kw in line.lower() for kw in ["copyright", "registration", "replaces", "directive", "unauthorized"]):
            continue
        if re.search(r"https?://", line):
            continue
        if re.search(r"\b(no\.|registration code|web[- ]?site)\b", line.lower()):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

def extract_intro_from_first_page(file_path: str) -> str:
    try:
        # Try with fitz (PyMuPDF)
        doc = fitz.open(file_path)
        if doc.page_count > 0:
            first_page_text = doc[0].get_text().strip()
            if first_page_text and not is_garbage(first_page_text) and not is_binary_or_junk(first_page_text):
                return clean_intro(first_page_text)
    except Exception as e:
        logger.warning(f"fitz failed for intro extraction: {e}")

   
    try:
        from pdf2image import convert_from_path
        from PIL import ImageEnhance, ImageFilter
        images = convert_from_path(file_path, dpi=400, first_page=1, last_page=1, poppler_path=POPPLER_PATH)
        if images:
            img = images[0].convert('L')
            img = ImageEnhance.Contrast(img).enhance(2.0)
            img = img.filter(ImageFilter.SHARPEN)
            text = pytesseract.image_to_string(img, config='--psm 6 --oem 3').strip()
            if text and not is_garbage(text) and not is_binary_or_junk(text):
                return clean_intro(text)
    except Exception as e:
        logger.warning(f"OCR fallback failed for intro extraction: {e}")

    # Nothing worked
    return ""

def ocr_image(image_path: str) -> str:
    try:
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"Invalid image path: {image_path}")
            return ""
        with Image.open(image_path) as img:
            img = img.convert('L')
            img = ImageEnhance.Contrast(img).enhance(2.0)
            img = img.filter(ImageFilter.SHARPEN)
            return pytesseract.image_to_string(img, config='--psm 6 --oem 3', lang='eng').strip()
    except Exception as e:
        logger.error(f"OCR failed for {image_path}: {e}")
        return ""

def full_page_ocr_fallback(file_path: str) -> List[Document]:  # Removed intro_text parameter
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            images = convert_from_path(
                file_path,
                dpi=400,
                output_folder=temp_dir,
                fmt='png',
                poppler_path=POPPLER_PATH
            )
            docs = []
            for i, img in enumerate(images):
                img = img.convert('L')
                img = ImageEnhance.Contrast(img).enhance(2.0)
                img = img.filter(ImageFilter.SHARPEN)
                text = pytesseract.image_to_string(img, config='--psm 6 --oem 3').strip()
                if text and not is_binary_or_junk(text):
                    docs.append(Document(
                        page_content=text,  # Don't add intro here
                        metadata={"filename": os.path.basename(file_path), "page": i + 1, "element_type": "fallback_ocr"}
                    ))
            return docs
    except Exception as e:
        logger.error(f"Full page OCR fallback failed: {e}")
        return []

def load_pdf_with_layout_analysis(file_path: str) -> List[Document]:  # Removed intro_text parameter
    filename = os.path.basename(file_path)
    logger.info(f"Using partition_pdf layout strategy for {filename}")
    try:
        elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,
            languages=["eng"],
            extract_ocr_text_from_image=True
        )
        docs = []
        for element in elements:
            try:
                text = ""
                if isinstance(element, Table):
                    text = str(element).strip()
                elif isinstance(element, ImageElement):
                    if hasattr(element.metadata, "image_path") and element.metadata.image_path:
                        text = ocr_image(element.metadata.image_path)
                else:
                    text = str(element).strip()
                
                if text and not is_binary_or_junk(text):
                    docs.append(Document(
                        page_content=text,  # Don't add intro here
                        metadata={"filename": filename, "element_type": "text"}
                    ))
            except Exception as inner_error:
                logger.warning(f"Error processing element in {filename}: {inner_error}")
                continue
        return docs
    except Exception as e:
        logger.error(f"partition_pdf failed for {filename}: {e}")
        return []

def split_docs_with_intro(documents: List[Document], intro_text: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Split documents and add intro text to each chunk"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    split_documents = splitter.split_documents(documents)
    
    # Add intro text to each chunk
    if intro_text:
        for doc in split_documents:
            doc.page_content = f"Topic: Atlas Copco SF 4 / SF 8  Scroll Air Compressors (Oil-Free) Series and data is for {intro_text}\n\nData : {doc.page_content}"
    
    return split_documents

# with semantic chunker
# def split_docs_with_intro(documents: List[Document], intro_text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[Document]:
#     """Split documents semantically and add intro text to each chunk"""
    
#     # Initialize SemanticChunker with your embedding model
#     embeddings = OpenAIEmbeddings()  # Replace with your actual embedding model
#     splitter = SemanticChunker(embeddings)

#     # Split the documents semantically
#     split_documents = splitter.split_documents(documents)
#     print("Splits are ")
#     print(split_documents[0])
#     # Prepend intro text to each chunk
#     if intro_text:
#         for doc in split_documents:
#             doc.page_content = f"Topic: {intro_text}\n\n{doc.page_content}"

#     return split_documents

def load_pdf_smart(file_path: str, chunk_size=1400, chunk_overlap=300) -> List[Document]:
    filename = os.path.basename(file_path)
    intro_text = extract_intro_from_first_page(file_path)
    # print("Got Intro Text ") 
    # print(intro_text)
    
    try:
        logger.info(f"Trying Fitz for {filename}")
        doc = fitz.open(file_path)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text()
            if text and not is_garbage(text) and not is_binary_or_junk(text):
                pages.append(Document(
                    page_content=text.strip(),  # Don't add intro here
                    metadata={"filename": filename, "page": i + 1}
                ))
        
        if pages:
            logger.info(f"Loaded with Fitz: {filename}")
            return split_docs_with_intro(pages, intro_text, chunk_size, chunk_overlap)
        else:
            raise ValueError("Fitz extracted no valid content")
    
    except Exception as e:
        logger.warning(f"Fitz failed: {e}")

    logger.info(f"Falling back to full-page OCR for {filename}")
    ocr_docs = full_page_ocr_fallback(file_path)  # Removed intro_text parameter
    if ocr_docs:
        logger.info(f"Full-page OCR succeeded for {filename}")
        return split_docs_with_intro(ocr_docs, intro_text, chunk_size, chunk_overlap)

    logger.error(f"Failed to process {filename} by any method.")
    return []