import os
import fitz
import logging
import pytesseract
import tempfile
from PIL import Image, ImageEnhance, ImageFilter
from typing import List
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Table, Image as ImageElement, CompositeElement
from pdf2image import convert_from_path

# Set up logger
logger = logging.getLogger("services.load_pdf")

# Set these paths to your local installs
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r'C:\Users\abc\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin'

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def is_garbage(text: str) -> bool:
    nonprintable = sum(1 for c in text if not c.isprintable())
    ratio = nonprintable / max(len(text), 1)
    return ratio > 0.4


def ocr_image(image_path: str) -> str:
    try:
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"Invalid image path: {image_path}")
            return ""

        with Image.open(image_path) as img:
            img = img.convert('L')
            img = ImageEnhance.Contrast(img).enhance(2.0)
            img = img.filter(ImageFilter.SHARPEN)
            return pytesseract.image_to_string(
                img, config='--psm 6 --oem 3', lang='eng'
            ).strip()

    except Exception as e:
        logger.error(f"OCR failed for {image_path}: {e}")
        return ""


def full_page_ocr_fallback(file_path: str) -> List[Document]:
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            images = convert_from_path(
                file_path,
                dpi=400,
                output_folder=temp_dir,
                fmt='png',
                poppler_path=POPPLER_PATH
            )

            full_text = []
            for img in images:
                img = img.convert('L')
                img = ImageEnhance.Contrast(img).enhance(2.0)
                img = img.filter(ImageFilter.SHARPEN)
                text = pytesseract.image_to_string(img, config='--psm 6 --oem 3')
                full_text.append(text)

            docs = []
            for i, img in enumerate(images):
                img = img.convert('L')
                img = ImageEnhance.Contrast(img).enhance(2.0)
                img = img.filter(ImageFilter.SHARPEN)

                text = pytesseract.image_to_string(img, config='--psm 6 --oem 3').strip()
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "filename": os.path.basename(file_path),
                            "page": i + 1,
                            "element_type": "fallback_ocr"
                        }
                    ))
            return docs

    except Exception as e:
        logger.error(f"Full page OCR fallback failed: {e}")
        return []


def load_pdf_with_layout_analysis(file_path: str) -> List[Document]:
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
                if isinstance(element, Table):
                    table_text = str(element).strip()
                    if table_text:
                        docs.append(Document(
                            page_content=f"TABLE:\n{table_text}",
                            metadata={"filename": filename, "element_type": "table"}
                        ))

                elif isinstance(element, ImageElement):
                    if hasattr(element.metadata, "image_path") and element.metadata.image_path:
                        text = ocr_image(element.metadata.image_path)
                        if text:
                            docs.append(Document(
                                page_content=text,
                                metadata={"filename": filename, "element_type": "image"}
                            ))

                else:
                    text = str(element).strip()
                    if text:
                        docs.append(Document(
                            page_content=text,
                            metadata={"filename": filename, "element_type": "text"}
                        ))

            except Exception as inner_error:
                logger.warning(f"Error processing element in {filename}: {inner_error}")
                continue

        return docs if docs else []
    except Exception as e:
        logger.error(f"partition_pdf failed for {filename}: {e}")
        return []


def split_docs(documents, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def load_pdf_smart(file_path: str, chunk_size=1500, chunk_overlap=300) -> List[Document]:
    filename = os.path.basename(file_path)

    try:
        logger.info(f"Trying Fitz for {filename}")
        doc = fitz.open(file_path)
        pages = []

        for i, page in enumerate(doc):
            text = page.get_text()
            if text and not is_garbage(text):
                pages.append(Document(
                    page_content=text.strip(),
                    metadata={"filename": filename, "page": i + 1}
                ))

        if pages:
            logger.info(f"Loaded with Fitz: {filename}")
            return split_docs(pages, chunk_size, chunk_overlap)
        else:
            raise ValueError("Fitz extracted no valid content")
    except Exception as e:
        logger.warning(f"Fitz failed: {e}")

    # Layout-based OCR fallback
    logger.info(f"Falling back to OCR layout for {filename}")
    layout_docs = load_pdf_with_layout_analysis(file_path)
    if layout_docs:
        logger.info(f"OCR layout succeeded for {filename}")
        return split_docs(layout_docs, chunk_size, chunk_overlap)

    # Full page OCR as last resort
    logger.info(f"Falling back to full-page OCR for {filename}")
    ocr_docs = full_page_ocr_fallback(file_path)
    if ocr_docs:
        logger.info(f"Full-page OCR succeeded for {filename}")
        return split_docs(ocr_docs, chunk_size, chunk_overlap)

    logger.error(f"Failed to process {filename} by any method.")
    return []
