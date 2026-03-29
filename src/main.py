from fastapi import FastAPI, UploadFile, File, Form
from .utils import load_document, chunk_text
from .rag_pipeline import ingest_document, query_rag

app = FastAPI(title="Local RAG API")

# ---- Upload documents ----
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    contents = await file.read()
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as f:
        f.write(contents)

    pages = load_document(temp_file)     
    all_chunks = []
    if file.filename.endswith(".pdf"):
        # all_chunks # (Do 3 different chunktypes)
        for page in pages:
            print("DOING TASK 1!!!!!!!")
            all_chunks = chunk_text(page[0])
            count = ingest_document(file.filename, all_chunks)
            print("DOING TASK 2!!!!!!!")
            all_chunks = chunk_text(page[1])
            count += ingest_document(file.filename, all_chunks, "image")
            print("DOING TASK 3!!!!!!!")
            all_chunks = chunk_text(page[2])
            count += ingest_document(file.filename, all_chunks, "table")
    else:
        for page in pages:
            all_chunks.extend(chunk_text(page))
        count = ingest_document(file.filename, all_chunks)
    return {"message": f"Ingested {count} chunks from {file.filename}"}

# ---- Query endpoint ----
@app.post("/query")
async def query(question: str = Form(...)):
    answer = query_rag(question)
    return {"answer": answer}
