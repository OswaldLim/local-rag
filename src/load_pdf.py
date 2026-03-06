from typing import List, Dict, Any
import io

import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def _ocr_image(img: Image.Image) -> str:
    """
    OCR text inside images (charts, diagrams, annotations).
    """
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    return pytesseract.image_to_string(gray)

def page_to_text_with_images(page: dict) -> str:
    """
    Combine text blocks, OCR from images, and table text into a single string.
    """
    text = " ".join([tb["text"] for tb in page["text_blocks"]])
    ocr_texts = " ".join([img["ocr_text"] for img in page["images"]])
    tables_text = " ".join([tbl["text"] for tbl in page.get("tables", [])])
    return [text, ocr_texts, tables_text]

# def page_to_text_with_images(page: dict) -> str:
#     """
#     Combine text blocks, OCR from images, and table text into a single string.
#     """
#     text = " ".join([tb["text"] for tb in page["text_blocks"]])
#     ocr_texts = " ".join([img["ocr_text"] for img in page["images"]])
#     tables_text = " ".join([tbl["text"] for tbl in page.get("tables", [])])
#     return text + " " + ocr_texts + " " + tables_text

def _extract_tables_from_page(page) -> List[Dict[str, Any]]:
    """
    Extract tables from a pdfplumber page as text blocks.
    """
    tables_data = []
    tables = page.extract_tables()
    for table in tables:
        # Flatten table to text
        table_text = " | ".join(["\t".join(cell if cell else "" for cell in row) for row in table])
        tables_data.append({"text": table_text})
    return tables_data

def load_pdf(file_path: str) -> List[str]:
    """
    Load a PDF and return structured, AI-ready page data:
    - text blocks with layout
    - images with OCR text
    - tables as text
    """

    pages: List[Dict[str, Any]] = []

    text_pdf = pdfplumber.open(file_path)
    image_pdf = fitz.open(file_path)

    try:
        for page_num in range(len(text_pdf.pages)):
            page_data: Dict[str, Any] = {
                "page": page_num,
                "text_blocks": [],
                "images": [],
                "tables": []
            }

            # ---------- TEXT (layout-aware) ----------
            page = text_pdf.pages[page_num]
            words = page.extract_words(use_text_flow=True)

            for w in words:
                page_data["text_blocks"].append({
                    "text": w["text"],
                    "bbox": (w["x0"], w["top"], w["x1"], w["bottom"])
                })

            # ---------- TABLES ----------
            page_data["tables"] = _extract_tables_from_page(page)

            # ---------- IMAGES ----------
            img_page = image_pdf[page_num]
            for img in img_page.get_images(full=True):
                xref = img[0]
                base_image = image_pdf.extract_image(xref)

                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                page_data["images"].append({
                    "image": image,
                    "bbox": img_page.get_image_bbox(img),
                    "ocr_text": _ocr_image(image)
                })

            pages.append(page_data)

    finally:
        text_pdf.close()
        image_pdf.close()

    # Return combined text (text + table + images OCR)
    return [page_to_text_with_images(page) for page in pages]