# from langchain_community.document_loaders import UnstructuredPDFLoader
from unstructured.partition.pdf import partition_pdf
from langchain_classic.schema import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import pandas as pd
from io import StringIO

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
    loader = UnstructuredPDFLoader(
        file_path=pdf_path,
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

    documents = loader.load()
    new_docs = []

    print(f"Loaded {len(documents)} documents(s) from PDF")
    for c in documents:
        print(type(c), c.page_content[:1000],"\n\n",c.metadata.get("category"),"\n")

    # for doc in documents:
    #     category = doc.metadata.get("category")

    #     if category == "Table":
    #         new_docs.append(format_table_as_single_doc(doc))
    #     else:
    #         new_docs.append(doc)

    # new_docs = combine_documents(new_docs)
    # print(len(new_docs))
    return new_docs

def create_summary():
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}

    """
    response = llm.invoke(prompt_text)

def format_table_as_single_doc(element):
    html_table = element.metadata.get("text_as_html")

    # ----------------------------
    # fallback (no HTML table)
    # ----------------------------
    if not html_table:
        return Document(
            page_content=element.page_content,
            metadata={**element.metadata, "source": "table"}
        )

    dfs = pd.read_html(StringIO(html_table))
    df = dfs[0]

    # ----------------------------
    # 1. FLATTEN COLUMN HEADERS
    # ----------------------------
    def clean_col(col):
        if isinstance(col, tuple):
            col = " ".join([c for c in col if "Unnamed" not in str(c)])
        col = str(col).strip()
        return None if "Unnamed" in col or col == "" else col

    df.columns = [clean_col(c) for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c is not None]]

    # ----------------------------
    # 2. SAFE CELL NORMALIZER
    # ----------------------------
    def normalize_cell(value):
        if isinstance(value, pd.Series):
            value = value.iloc[0] if len(value) > 0 else None

        if isinstance(value, (list, tuple, dict)):
            value = str(value)

        if pd.isna(value):
            return None

        value = str(value).strip()
        return value if value else None

    # ----------------------------
    # 3. BUILD LLM-FRIENDLY TEXT
    # ----------------------------
    table_name = element.metadata.get("table_name", "TABLE")

    lines = []
    lines.append(f"Table: {table_name}")
    lines.append("This table contains structured technical specifications.")
    lines.append("")

    # schema overview
    lines.append("Fields:")
    lines.append(", ".join(df.columns.astype(str)))
    lines.append("")

    # ----------------------------
    # 4. ROW SERIALIZATION
    # ----------------------------
    for i, row in df.iterrows():
        row_parts = []

        for col in df.columns:
            value = normalize_cell(row[col])

            if value is None:
                continue

            row_parts.append(f"{col}: {value}")

        if not row_parts:
            continue

        lines.append(f"Entry {i + 1}: " + "; ".join(row_parts))

    content = "\n".join(lines)

    # ----------------------------
    # 5. RETURN DOCUMENT
    # ----------------------------
    return Document(
        page_content=content,
        metadata={
            **element.metadata,
            "source": "table",
            "rows": len(df),
            "columns": list(df.columns),
            "table_id": element.metadata.get("element_id"),
        }
    )

if __name__ == "__main__":
    print("start")
    load_pdf("test_files\\2. Medium Pressure Accel Valves-installation.pdf")