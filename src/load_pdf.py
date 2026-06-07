# from langchain_community.document_loaders import UnstructuredPDFLoader
from unstructured.partition.pdf import partition_pdf
from langchain_classic.schema import Document
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import uuid
import base64

llm = OllamaLLM(
    model="llama3.2",
    base_url="http://ollama:11434"
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
    )

    print(f"Loaded {len(chunks)} documents(s) from PDF")

    tables = []
    texts = []

    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):  # Check if it's a CompositeElement
            for element in chunk.metadata.orig_elements:  # Iterate through its elements
                if "Table" in str(type(element)):  # Now check for Table type
                    tables.append(element)  # Append the table element
            texts.append(chunk)  # Still append the CompositeElement to texts

    images = get_images_base64(chunks)

    text_summary, table_summary, tables_html = create_summary(texts=texts, tables=tables)
    image_summary = summarize_image(images=images)

    return format_to_document(texts, text_summary, tables_html=tables_html, table_summaries=table_summary, image_summaries= image_summary, images=images)

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

def format_to_document(texts, text_summaries, tables_html, table_summaries, images, image_summaries):
    # 1) Make flat Documents for each modality (page_content = summary; metadata keeps originals)
    docs = []

    # text
    for original, summary in zip(texts, text_summaries):
        docs.append(Document(
            page_content=summary,
            metadata={
                "id": str(uuid.uuid4()),
                "modality": "text",
                "original": original.page_content if hasattr(original, "page_content") else str(original)
            }
        ))

    # tables
    for original_html, summary in zip(tables_html, table_summaries):
        docs.append(Document(
            page_content=summary,
            metadata={
                "id": str(uuid.uuid4()),
                "modality": "table",
                "original": original_html
            }
        ))

    # images (store the base64 so we can attach it later if needed)
    for b64, summary in zip(images, image_summaries):
        docs.append(Document(
            page_content=summary,   # image summary text
            metadata={
                "id": str(uuid.uuid4()),
                "modality": "image",
                "image_b64": b64
            }
        ))

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
    model = ChatOllama(model="llama3.2-vision", temperature=0)

    prompt_template = """Describe the image in detail. For context, 
                    the image is part of a research paper explaining the transformers 
                    architecture. Be specific about graphs, such as bar plots."""

    # 2. Setup the prompt
    # Note: Llama 3.2 Vision expects the image format within the message structure
    prompt = ChatPromptTemplate.from_messages([
        ("user", [
            {"type": "text", "text": prompt_template},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}}
        ])
    ])

    # 3. Create the chain
    chain = prompt | model | StrOutputParser()

    # 4. Run the batch
    # Pass a list of dictionaries with the key 'image' containing your base64 strings
    image_summaries = chain.batch([{"image": img} for img in images])

    return image_summaries

def create_summary(tables, texts):
    # Prompt
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}

    """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    model = ChatOllama(temperature=0.5, model="llama3.2")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 1})
    tables_html = [table.metadata.text_as_html for table in tables]
    table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 2})

    return text_summaries, table_summaries, tables_html





if __name__ == "__main__":
    print("start")
    doc = load_pdf("test_files\\2. Medium Pressure Accel Valves-installation.pdf")
    print(f"\n\n\n\n\n{type(doc)}")