from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from .utils import load_document, chunk_text
from .rag_pipeline import ingest_document, query_rag
import uuid

app = FastAPI(title="Local RAG API")

tasks = {}

# ---- Upload documents ----
@app.post("/ingest")
async def ingest(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    tasks[task_id] = "processing"
    contents = await file.read()
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as f:
        f.write(contents)

    background_tasks.add_task(process_ingestion_task, task_id, temp_file)
    return {"task_id": task_id, "status": "Processing started in background"}

# ---- Query endpoint ----
@app.post("/query")
async def query(question: str = Form(...)):
    answer = query_rag(question)
    return {"answer": answer}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    status = tasks.get(task_id, "not_found")
    return {"task_id": task_id, "status": status}

def process_ingestion_task(task_id, temp_file):
    pages = load_document(temp_file)     
    all_chunks = []
    if temp_file.endswith(".pdf"):
        count = ingest_document(pages)
    else:
        for page in pages:
            all_chunks.extend(chunk_text(page))
        count = ingest_document(temp_file[:4], all_chunks)
    tasks[task_id] = "completed"
    return {"message": f"Ingested {count} chunks from {temp_file[:4]}"}