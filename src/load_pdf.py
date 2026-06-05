from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_classic.schema import Document
import pandas as pd
from io import StringIO


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
        pdf_path,
        mode="elements",
        strategy="hi_res",
        infer_table_structure=True
    )

    documents = loader.load()
    new_docs = []

    print(f"Loaded {len(documents)} documents(s) from PDF")

    for doc in documents:
        category = doc.metadata.get("category")

        if category == "Table":
            new_docs.append(format_table_as_single_doc(doc))
        else:
            new_docs.append(doc)

    new_docs = combine_documents(new_docs)
    print(len(new_docs))
    return new_docs

def format_table_as_single_doc(element):
    html_table = element.metadata.get("text_as_html")

    if not html_table:
        # fallback
        return Document(
            page_content=element.page_content,
            metadata={**element.metadata, "source": "table", "text": element.page_content}
        )

    dfs = pd.read_html(StringIO(html_table))
    df = dfs[0]

    # Convert entire table into structured text
    table_text = []

    table_text.append("TABLE CONTENT:")

    # Add header row
    headers = " | ".join(str(col) for col in df.columns)
    table_text.append(headers)
    table_text.append("-" * len(headers))

    # Add rows
    for _, row in df.iterrows():
        row_text = " | ".join(str(row[col]) for col in df.columns)
        table_text.append(row_text)

    content = "\n".join(table_text)

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