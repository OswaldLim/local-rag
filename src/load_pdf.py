# from langchain_community.document_loaders import UnstructuredPDFLoader
from unstructured.partition.pdf import partition_pdf
from langchain_classic.schema import Document
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from itertools import batched
import asyncio
import uuid
import base64
from PIL import Image
import io

import time

URL = "http://ollama:11434"

llm = OllamaLLM(
    model="llama3.2",
    base_url=URL
)

# Main function for loading pdf
async def load_pdf(pdf_path):
    # 1. Use lazy partitioning (this is already a generator)
    chunks = partition_pdf(
        filename=pdf_path,
        mode="elements",
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
        lazy=True
    )

    print(len(chunks))

    # 2. Process element-by-element
    for chunk in chunks:
        # --- HANDLE TABLES ---
        if "Table" in str(type(chunk)):
            # Process table in real-time
            summary, html = await process_table_async(chunk)
            yield format_as_document(chunk, summary=summary, html=html, modality="table", filepath=pdf_path)
            
        # --- HANDLE COMPOSITE ELEMENTS (Text + Nested Tables) ---
        # --- HANDLE COMPOSITE ELEMENTS (Text + Nested Images/Tables) ---
        elif "CompositeElement" in str(type(chunk)):
            # 1. Yield the text part of the composite first
            yield format_as_document(chunk, modality="text", filepath=pdf_path)
            
            # 2. Dig into the nested elements
            if hasattr(chunk.metadata, "orig_elements"):
                for element in chunk.metadata.orig_elements:
                    el_type = str(type(element))
                    
                    if "Table" in el_type:
                        summary, html = await process_table_async(element)
                        yield format_as_document(element, summary=summary, html=html, modality="table", filepath=pdf_path)
                        
                    elif "Image" in el_type:
                        # Extract B64 from the nested element
                        b64 = element.metadata.image_base64
                        summary = await summarize_single_image(b64)
                        yield format_as_document(element, summary=summary, b64=b64, modality="image", filepath=pdf_path)

# Image Handling Functions Below
def resize_if_large(b64_string, threshold_kb=5):
    # 1. Remove prefix if present
    header, encoded = b64_string.split(",", 1) if "," in b64_string else ("", b64_string)
    
    # 2. Check size (5KB = 5120 bytes)
    size_bytes = (len(encoded) * 3) // 4
    if size_bytes < (threshold_kb * 1024):
        return b64_string # Return original if small

    # 3. Resize if too big
    img_data = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_data))
    
    # Resize keeping aspect ratio
    img.thumbnail((1024, 1024)) 
    
    # 4. Convert back to base64
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    new_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return f"{header},{new_b64}" if header else new_b64

def filter_resize_image(image):
    # 1. Calculate size in KB
    size_kb = get_base64_image_size_bytes(image) / 1024
    
    # 2. Filter: Only keep images > 2KB (assuming these are meaningful)
    if size_kb > 2.5:
        # 3. Resize: Pass through the resize function if it's "large"
        # (e.g., > 50KB or whatever threshold you choose)
        processed_img = resize_if_large(image, threshold_kb=50)
        return (processed_img)
    return None    
    

def get_base64_image_size_bytes(b64_string):
    # If your string has a prefix like "data:image/jpeg;base64,", remove it first
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
        
    # Calculate size
    return (len(b64_string) * 3) // 4 - b64_string.count('=', -2)

async def summarize_single_image(image_b64: str) -> str:
    """Processes a single image base64 string."""
    model = ChatOllama(model="moondream", temperature=0, base_url=URL)
    
    if filter_resize_image(image_b64) == None:
        return "Image too small"

    prompt = ChatPromptTemplate.from_messages([
        ("user", [
            {"type": "text", "text": "Describe the image in detail. Overview and key details. Under 200 words."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ])
    ])
    
    chain = prompt | model | StrOutputParser()
    return await chain.ainvoke({})

# Text and Table Handling
async def process_table_async(table_element) -> tuple[str, str]:
    """Summarizes a single table element."""
    html = table_element.metadata.text_as_html
    
    prompt = ChatPromptTemplate.from_template(
        "Summarize this table, summarizing each row of data into a sentence. "
        "Return only the summary. Table content: {element}"
    )
    
    model = ChatOllama(temperature=0.5, model="llama3.2", base_url=URL)
    chain = prompt | model | StrOutputParser()
    
    # Process immediately
    summary = await chain.ainvoke({"element": html})
    return summary, html

# Used for formatting documents
def format_as_document(element, summary=None, html=None, b64=None, modality="text", filepath=""):
    content = summary if summary else (element.page_content if hasattr(element, "page_content") else str(element))
    original = html if html else ("N/A" if b64 else (element.page_content if hasattr(element, "page_content") else str(element)))
    
    return Document(
        page_content=content,
        metadata={
            "id": str(uuid.uuid4()),
            "modality": modality,
            "original": original,
            "filename": filepath
        }
    )
# Local Testing code
async def main():
    start_time = time.perf_counter()
    print("start")
    
    file_path = "test_files\\2. Medium Pressure Accel Valves-installation.pdf"

    # 1. Capture the generator object
    doc_generator = load_pdf(file_path)
    
    count = 0
    # 2. Consume the generator stream
    async for doc in doc_generator:
        count+=1
        # Process or print individual docs as they arrive
        print(f"Ingested chunk: {doc.metadata['modality']} - {doc.page_content[:50]}...")
    
    endtime = time.perf_counter()
    print(f"FINISHED: {endtime - start_time} with count {count}")

if __name__ == "__main__":
    asyncio.run(main())