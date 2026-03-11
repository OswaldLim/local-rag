from typing import List, Dict, Any
import io

import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import cv2
import numpy as np
import logging
import re
from difflib import SequenceMatcher

# Get the standard logger
logger = logging.getLogger("app")
logging.basicConfig(level=logging.INFO)

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


table_setting = {
    "vertical_strategy": "lines", 
    "horizontal_strategy": "lines",
    "edge_min_length": 4,
    "intersection_tolerance": 3.8,
    "snap_tolerance": 10,
    "join_tolerance": 10,
}

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
    tables = page.extract_tables(table_settings=table_setting)

    # check_tables = page.find_tables(table_settings=table_setting)
    # for tab in check_tables:
    #     print(f"checking table cells\n {tab.cells}\n")
    #     print(f"checking table rows\n {tab.rows}\n\n\n\n\n\n")

    for table in tables:
        if not table or len(table) < 2: continue
        
        # 1. Identify headers
        top_headers = [cell.strip() if cell else "" for cell in table[0]]
        semantic_rows = []
        
        # 2. Iterate through data rows (starting from index 1)
        for row in table[1:]:
            # The first element of the row is our "Side Header" (e.g., 'Workers')
            side_header = row[0].strip() if row[0] else "Row"
            
            row_content = []
            # 3. Pair each cell with its Top Header
            for i, cell in enumerate(row[1:], start=1):
                header = top_headers[i] if i < len(top_headers) else f"Col_{i}"
                value = cell.strip() if cell else "N/A"
                
                # Format: [SideHeader, TopHeader]: Value
                row_content.append(f"[{side_header}, {header}]: {value}")
            
            semantic_rows.append(" | ".join(row_content))
            
        tables_data.append({"text": " || ".join(semantic_rows)})
    logger.info(f"Extracted table data:  {tables_data}")
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
            words = page.extract_words(use_text_flow=True, x_tolerance=2, extra_attrs=["fontname"])

            for w in words:
                page_data["text_blocks"].append({
                    "text": w["text"],
                    "bbox": (w["x0"], w["top"], w["x1"], w["bottom"])
                })

            # ---------- TABLES ----------
            page_data["tables"] = _extract_tables_from_page(page)

            # ---------- IMAGES ----------
            if len(page_data["text_blocks"] <= 0):
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
            # print(f"page data: {page_data}", flush=True)
            pages.append(page_data)
            logger.info(f"Page table data {page_data["tables"]}\n\n")

    finally:
        print(len(pages))
        text_pdf.close()
        image_pdf.close()

    # Return combined text (text + table + images OCR)
    return [page_to_text_with_images(page) for page in pages]

def normalize(text: str) -> str:
    """
    Normalize text for deduplication.
    - lowercase
    - remove extra whitespace
    - remove repeated punctuation spacing
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)   # collapse whitespace
    text = text.strip()
    return text

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

