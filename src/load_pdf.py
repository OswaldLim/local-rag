# from langchain_community.document_loaders import UnstructuredPDFLoader
from unstructured.partition.pdf import partition_pdf
from langchain_classic.schema import Document
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from itertools import batched
import asyncio
from ollama import AsyncClient
import uuid
import base64
from PIL import Image
import io

import time

URL = "http://localhost:11434"

llm = OllamaLLM(
    model="llama3.2",
    base_url=URL
)


def combine_documents(documents):
    combined = []
    buffer = []
    current_metadata = None

    def flush():
        nonlocal buffer, current_metadata
        if not buffer:
            return

        text = "\n".join(buffer).strip()
        if text:
            combined.append(
                Document(
                    page_content=text,
                    metadata=current_metadata or {}
                )
            )

        buffer = []
        current_metadata = None

    for doc in documents:
        category = doc.metadata.get("category", "")
        text = doc.page_content.strip()

        # TABLES: keep standalone
        if doc.metadata.get("source") == "table" or category == "Table":
            flush()
            combined.append(doc)
            continue

        # HEADER/TITLE: start new block
        if category in ["Title", "Header"]:
            flush()
            buffer.append(text)
            current_metadata = doc.metadata
            continue

        # NORMAL TEXT
        if not buffer:
            current_metadata = doc.metadata

        buffer.append(text)

    flush()
    return combined

async def load_pdf(pdf_path):
    chunks = partition_pdf(
        filename=pdf_path,
        mode="elements",
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image", "Table"],   # Add 'Table' to list to extract image of tables
        extract_image_block_to_payload=True,   # if true, will extract base64 for API usage
        chunking_strategy="by_title",          # or 'basic'
        max_characters=10000,                  # defaults to 500
        combine_text_under_n_chars=2000,       # defaults to 0
        new_after_n_chars=6000,
        lazy = True
    )

    tables = []
    texts = []

    for chunk in chunks:
        # print(str(type(chunk)))
        if "CompositeElement" in str(type(chunk)):  # Check if it's a CompositeElement
            # print(f"\n\nInside\n{chunk.metadata.orig_elements}\n")
            for element in chunk.metadata.orig_elements:  # Iterate through its elements
                # print(f"\n\nHIIIIIIII{element}\n\n")
                if "Table" in str(type(element)):  # Now check for Table type
                    # print(f"\n\ntable element: {element}\n\n")
                    tables.append(element)  # Append the table element
            texts.append(chunk)  # Still append the CompositeElement to texts
        elif "Table" in str(type(chunk)):
            tables.append(chunk)


    images = get_images_base64(chunks)
    filtered_images = filter_resize_image(images)

    table_summary, tables_html = create_summary(tables=tables)
    # print(table_summary, "\n\n")
    # print(tables_html, "\n\n")
    image_summary = []
    
    async for summary in summarize_image(filtered_images):
        image_summary.append(summary)
        print(f"Received summary: {summary[:30]}...")

    # image_summary = []

    return format_to_document(texts, tables_html=tables_html, table_summaries=table_summary, image_summaries= image_summary, images=images, filepath = pdf_path)

def display_base64_image(b64_string):
    import io
    from PIL import Image

    if ',' in b64_string:
        b64_string = b64_string.split(',', 1)[1]
        
    # Decode and open the image
    image_data = base64.b64decode(b64_string)
    image = Image.open(io.BytesIO(image_data))
    
    # Display the image
    image.show()

def format_to_document(texts, tables_html, table_summaries, images, image_summaries, filepath):
    # 1) Make flat Documents for each modality (page_content = summary; metadata keeps originals)
    docs = []

    # text
    for original in texts:
        docs.append(Document(
            page_content=original.page_content if hasattr(original, "page_content") else str(original),
            metadata={
                "id": str(uuid.uuid4()),
                "modality": "text",
                "original": original.page_content if hasattr(original, "page_content") else str(original),
                "filename":filepath
            }
        ))

    # tables
    for original_html, summary in zip(tables_html, table_summaries):
        # print("\n\nAppending Table\n\n")
        docs.append(Document(
            page_content=summary,
            metadata={
                "id": str(uuid.uuid4()),
                "modality": "table",
                "original": original_html,
                "filename":filepath
            }
        ))

    # images (store the base64 so we can attach it later if needed)
    for b64, summary in zip(images, image_summaries):
        docs.append(Document(
            page_content=summary,   # image summary text
            metadata={
                "id": str(uuid.uuid4()),
                "modality": "image",
                "image_b64": b64,
                "filename":filepath
            }
        ))

    # print(len(docs))

    return docs


def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

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

def filter_resize_image(images):
    filtered_images = []
    
    for img in images:
        # 1. Calculate size in KB
        size_kb = get_base64_image_size_bytes(img) / 1024
        
        # 2. Filter: Only keep images > 2KB (assuming these are meaningful)
        if size_kb > 2.5:
            # 3. Resize: Pass through the resize function if it's "large"
            # (e.g., > 50KB or whatever threshold you choose)
            processed_img = resize_if_large(img, threshold_kb=50)
            filtered_images.append(processed_img)
            
    return filtered_images

def get_base64_image_size_bytes(b64_string):
    # If your string has a prefix like "data:image/jpeg;base64,", remove it first
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
        
    # Calculate size
    return (len(b64_string) * 3) // 4 - b64_string.count('=', -2)

async def summarize_image(images):
    model = ChatOllama(model="llava", temperature=0, base_url=URL)

    prompt_template = """
        Describe the image in detail. 
        Follow this structure: 
        1. Overview: What is this image? 
        2. Details: Key trends, data points, or labels (use bullet points). 
        Keep the total response under 200 words and be concise.
        """

    print(len(images))

    # 2. Setup the prompt
    # Note: Llama 3.2 Vision expects the image format within the message structure
    prompt = ChatPromptTemplate.from_messages([
        ("user", [
            {"type": "text", "text": prompt_template},
            # Ensure the variable name inside the URL matches the key in your input dict
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}}
        ])
    ])


    # 3. Create the chain
    chain = prompt | model | StrOutputParser()

    semaphore = asyncio.Semaphore(4)

    async def _process_single(img):
        async with semaphore:
            return await chain.ainvoke({"image": img})
        
    tasks = [_process_single(img) for img in images]

    for completed_task in asyncio.as_completed(tasks):
        result = await completed_task
        yield result

    # 4. Run the batch
    # Pass a list of dictionaries with the key 'image' containing your base64 strings
    # image_summaries = []
    # for index, img in enumerate(images):
    #     print(f"--- Processing image {index + 1} ---")
        
    #     try:
    #         # invoke() processes one image at a time
    #         result = chain.invoke({"image": img})
    #         image_summaries.append(result)
            
    #         print(f"Successfully processed image {index + 1}.")
    #         print(f"Summary: {result[:50]}...") # Preview the summary
            
    #     except Exception as e:
    #         print(f"Error processing image {index + 1}: {e}")
    #         image_summaries.append(None)

    # parallel batching
    # image_summaries = await chain.abatch(
    #     [{"image": img} for img in images],
    #     config={'max_concurrency': 4}
    # )

    # print(f"\n\n{image_summaries}\n\n")

    # # image_summaries = chain.batch([{"image": img} for img in images], config={'max_concurrency': 1})

    # print("finish summarising images")
    # return image_summaries

def create_summary(tables):
    # Prompt
    prompt_text = """
    You are an assistant tasked with summarizing tables.
    When summarising tables summarise each row of data into a sentence.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table chunk: {element}

    """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    model = ChatOllama(temperature=0.5, model="llama3.2", base_url=URL)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    table_summaries = []
    tables_html = [table.metadata.text_as_html for table in tables]
    for batch in batched(tables_html, 10):
        table_summaries.extend(summarize_chain.batch(list(batch), {"max_concurrency": 4}))

    print("done summarising")

    return table_summaries, tables_html





if __name__ == "__main__":
    global start_time
    start_time = time.perf_counter()
    print("start")
    # file_path = "test_files\\PDS_X30 FHMs_C_UnLck_MKT-0044.pdf"
    file_path = "test_files\\2. Medium Pressure Accel Valves-installation.pdf"
    image_path =  "C:\\Users\\Lenovo\\Desktop\\SCHOOL\\Personal\\Others\\profile_new.jpg"

    # def image_to_base64(image_path):
    #     with open(image_path, "rb") as image_file:
    #         # Read the binary data
    #         binary_data = image_file.read()
    #         # Encode to base64
    #         base64_encoded_data = base64.b64encode(binary_data)
    #         # Convert to string and decode to UTF-8
    #         base64_string = base64_encoded_data.decode('utf-8')
    #         return base64_string

    # image_64 = image_to_base64(image_path)
    # print(f"Finished converting image: {image_64}")
    # output = summarize_image([image_64])
    # print(output[0])

    docs = asyncio.run(load_pdf(file_path))
    endtime = time.perf_counter()
    print(f"FINISHED: {endtime - start_time}")
    # for doc in docs:
    #     print(f"{doc.page_content}, {doc.metadata}\n\n")

