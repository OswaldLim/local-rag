import os
import tempfile
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredPDFLoader
)
import pytesseract

# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def load_pdf(pdf_path):
    loader = UnstructuredPDFLoader(pdf_path,"elements",strategy="hi_res")
    documents = loader.load()

    print(f"Loaded {len(documents)} documents(s) from PDF")
    for i, doc in enumerate(documents):
            if i > 30:
                break
            print(f"\nDocument {i+1} Content Preview: {doc.page_content[:100]}")
            print(f"\nMetadata: {doc.metadata}")


if __name__ == "__main__":
    print("Start")
    load_pdf("test_files\\2. Medium Pressure Accel Valves-installation.pdf")