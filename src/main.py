from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
from .utils import load_document, chunk_text
from .rag_pipeline import ingest_with_buffer, stream_rag_response
from .telegram_utils import stream_to_telegram
from pydantic import BaseModel
import uuid

class QueryRequest(BaseModel):
    chat_id: int
    draft_id: int
    question: str

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
async def query(request: QueryRequest):
    print("Started streaming")
    text = await stream_to_telegram(
        chat_id=request.chat_id, 
        draft_id=request.draft_id, 
        llm_stream=stream_rag_response(request.question)
    )

    return {"answer": text}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    status = tasks.get(task_id, "not_found")
    if status == "failed":
        return {"task_id": task_id, "status": status}
    return {"task_id": task_id, "status": status}

async def process_ingestion_task(task_id, temp_file):
    try:
        count = 0
        doc_stream = load_document(temp_file)

        print("Finish loading document")

        async for count in ingest_with_buffer(doc_stream):
            print(f"Ingestion finished. Total: {count}")
        tasks[task_id] = "completed"
        return {"message": f"Successfully ingested {count} chunks."}
    except Exception as e:
        tasks[task_id] = "failed"
        print(str(e))
        return {"error": str(e)}