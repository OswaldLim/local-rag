# from langchain_community.document_loaders import UnstructuredPDFLoader
from unstructured.partition.pdf import partition_pdf
from langchain_classic.schema import Document
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from itertools import batched
import uuid
import base64

import time

llm = OllamaLLM(
    model="llama3.2",
    base_url="http://localhost:11434"
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

def load_pdf(pdf_path):
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


    print(f"Loaded {len(chunks)} documents(s) from PDF")
    global endtime1
    endtime1 = time.perf_counter()
    elapsed_time = endtime1 - start_time
    print(f"The code took {elapsed_time:.4f} seconds to run.")
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

    table_summary, tables_html = create_summary(tables=tables)
    # print(table_summary, "\n\n")
    # print(tables_html, "\n\n")
    image_summary = summarize_image(images=images)
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

def summarize_image(images):
    model = ChatOllama(model="llava", temperature=0, base_url="http://localhost:11434")

    prompt_template = """Describe the image in detail. Be specific about graphs, such as bar plots."""

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

    def debug_print(x):
        print(f"DEBUG DATA: {x}")
        global start_time
        endtime3 = time.perf_counter()
        elapsed_time = endtime3 - start_time
        print(f"\n\nThe code took {elapsed_time:.4f} seconds to run.")
        return x

    def debug_finish(x):
        print("finish")
        return x

    # 3. Create the chain
    # chain = prompt | RunnableLambda(debug_print) | model | RunnableLambda(debug_finish) | StrOutputParser()
    chain = prompt | model | StrOutputParser()


    # 4. Run the batch
    # Pass a list of dictionaries with the key 'image' containing your base64 strings
    image_summaries = []
    for index, img in enumerate(images):
        print(f"--- Processing image {index + 1} ---")
        
        try:
            # invoke() processes one image at a time
            result = chain.invoke({"image": img})
            image_summaries.append(result)
            
            print(f"Successfully processed image {index + 1}.")
            print(f"Summary: {result[:50]}...") # Preview the summary
            
        except Exception as e:
            print(f"Error processing image {index + 1}: {e}")
            image_summaries.append(None)

    # image_summaries = chain.batch([{"image": img} for img in images], config={'max_concurrency': 1})

    print("finish summarising images")
    global endtime3
    endtime3 = time.perf_counter()
    elapsed_time = endtime3 - start_time
    print(f"The code took {elapsed_time:.4f} seconds to run.")
    return image_summaries

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
    # model = ChatOllama(temperature=0.5, model="llama3.2", base_url="http://ollama:11434")
    model = ChatOllama(temperature=0.5, model="llama3.2", base_url="http://localhost:11434")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    table_summaries = []
    tables_html = [table.metadata.text_as_html for table in tables]
    for batch in batched(tables_html, 10):
        table_summaries.extend(summarize_chain.batch(list(batch), {"max_concurrency": 5}))

    print("done summarising")
    global endtime2
    endtime2 = time.perf_counter()
    elapsed_time = endtime2 - start_time
    print(f"The code took {elapsed_time:.4f} seconds to run.")
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

    docs = load_pdf(file_path)
    for doc in docs:
        print(f"{doc.page_content}, {doc.metadata}\n\n")

