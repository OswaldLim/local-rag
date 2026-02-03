# Local RAG Pipeline with FastAPI, Docker, and n8n

This project is a local Retrieval-Augmented Generation (RAG) pipeline using FastAPI for the API, Docker for containerization, and n8n for automation. It supports multiple document types including PDF, Word, Excel, CSV, TXT, and PPTX.

---

Features:

- Document ingestion: PDF, DOCX, TXT, CSV, XLSX, PPTX
- RAG-ready text chunking
- Query endpoint: Ask questions about ingested documents
- Dockerized environment: Run Python API, Ollama server, Qdrant, and n8n locally
- n8n automation: Automate uploads and queries
- Virtual environment support: Keep Python dependencies isolated

---

Folder Structure:

rag-project/
- venv/               # Python virtual environment (ignored by Git)
- src/                # Source code (FastAPI, RAG, utils)
  - main.py           # FastAPI app
  - rag_pipeline.py   # RAG ingestion and query functions
  - utils.py          # Helpers (chunking, document loaders, etc.)
- requirements.txt    # Python dependencies
- docker-compose.yml  # Docker services: Python API, n8n, Qdrant, Ollama
- Dockerfile.python   # Python API container
- README.md           # Project documentation

---

Setup Instructions:

1. Clone the repository

   git clone https://github.com/username/rag-project.git
   cd rag-project

2. Docker setup

   Ensure Docker is installed and running.

   Build and start containers:

   docker-compose up -d --build

   Services included:

   - python-api → FastAPI backend (http://localhost:8000)
   - n8n → Automation interface (http://localhost:5678)
   - qdrant → Vector database (http://localhost:6333)
   - ollama → Local LLM server (http://localhost:11434)

---

Usage:

1. Open n8n at (http://localhost:5678)

2. Ingest Documents
Supported file types:

- PDF
- DOCX / DOC
- TXT
- CSV
- XLSX / XLS
- PPTX / PPT

2. Query documents
type in n8n chat

Response:

   {"answer": "Predicates are ..."}

3. n8n automation

- Open n8n at http://localhost:5678
- Create workflows that upload files to /ingest and query /query endpoint
- n8n can handle binary file uploads automatically

---

Notes:

- Virtual environment: Do not commit venv/ to Git; add it to .gitignore
- Ollama models: Must be pulled locally if using a local LLM
- Docker volumes: Persistent data is stored in volumes (ollama_data, qdrant_data, n8n_data)
- File conversions: .doc → .docx and .ppt → .pptx if needed

---

Recommended Libraries:

File Type | Library
PDF       | PyPDF2
DOCX      | python-docx
TXT       | built-in open()
CSV       | pandas
XLSX      | pandas + openpyxl
PPTX      | python-pptx