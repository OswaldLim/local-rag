import os
import tempfile
from pathlib import Path
import pandas as pd
from io import StringIO
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredPDFLoader
)
import pytesseract

# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def load_pdf(pdf_path):
    loader = UnstructuredPDFLoader(pdf_path,"elements",strategy="hi_res", infer_table_structure=True)
    documents = loader.load()

    print(f"Loaded {len(documents)} documents(s) from PDF")
    print(f"{documents[12].page_content}\n")
    # print(documents[12])
    format_table(documents[12])
    # for i, doc in enumerate(documents):
    #         if i > 30:
    #             break
    #         if i == 12:
    #             print(f"\n{doc.page_content}")
    #         else:
    #             print(f"\nDocument {i+1} Content Preview: {doc.page_content[:100]}")
    #         print(f"\nMetadata: {doc.metadata}")

def format_table(element):
    print(element.metadata)
    html_table = element.metadata["text_as_html"]
        
    # 3. Convert HTML string to a list of DataFrames
    # read_html returns a list, so we take the first element [0]
    dfs = pd.read_html(StringIO(html_table))
    df = dfs[0]
    
    print(df.head())

if __name__ == "__main__":
    print("Start")
    load_pdf("test_files\\2. Medium Pressure Accel Valves-installation.pdf")