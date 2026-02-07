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
    text = " ".join([tb["text"] for tb in page["text_blocks"]])
    ocr_texts = " ".join([img["ocr_text"] for img in page["images"]])
    return text + " " + ocr_texts


def load_pdf(file_path: str) -> List[str]:
    """
    Load a PDF and return structured, AI-ready page data:
    - text blocks with layout
    - images with OCR text
    """

    pages: List[Dict[str, Any]] = []

    text_pdf = pdfplumber.open(file_path)
    image_pdf = fitz.open(file_path)

    try:
        for page_num in range(len(text_pdf.pages)):
            page_data: Dict[str, Any] = {
                "page": page_num,
                "text_blocks": [],
                "images": []
            }

            # ---------- TEXT (layout-aware) ----------
            page = text_pdf.pages[page_num]
            words = page.extract_words(use_text_flow=True)

            for w in words:
                page_data["text_blocks"].append({
                    "text": w["text"],
                    "bbox": (w["x0"], w["top"], w["x1"], w["bottom"])
                })

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

    return [page_to_text_with_images(page) for page in pages]
