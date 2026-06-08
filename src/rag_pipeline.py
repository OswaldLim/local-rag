from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_qdrant import QdrantVectorStore
import uuid

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://ollama:11434"
)

llm = OllamaLLM(
    model="llama3.2",
    base_url="http://ollama:11434",
    temperature=0.5
)

qdrant = QdrantClient(host="qdrant", port=6333)
COLLECTION_NAME = "rag_docs"

# qdrant.delete_collection(collection_name=COLLECTION_NAME)

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": 768, "distance": "Cosine"},
        timeout=30
    )

vector_store = QdrantVectorStore(
    client=qdrant,
    collection_name=COLLECTION_NAME,
    embedding=embedding_model
)

def ingest_document(text_chunks: list):
    points = []
    sparse_points = []
    print(f"INGESTING DOCUMENTSSSS\n  {text_chunks}", flush=True)
    if len(text_chunks) == 0:
        return 0
    for i, chunk in enumerate(text_chunks):
        vector = embedding_model.embed_query(chunk.page_content)
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": chunk.page_content,
                **chunk.metadata
            }
        ))
        sparse_points.append
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)

def query_rag(query: str, top_k: int = 10) -> str:
    # Embed query
    query_vector = embedding_model.embed_query(query)
    result = qdrant.query_points(
            collection_name=COLLECTION_NAME, 
            query=query_vector,
            limit=top_k)
    print("FINISH QUERYing points!!!!!")

    # New version
    context = format_context(result.points)

    print(f"ContextSTTTTT\n\n{context}\n\n")

    prompt = f"""
    You are an expert assistant. Answer the question using ONLY the context provided below.
    
    Context:
    {context}
    
    Question: {query}
    
    Instructions:
    1. If the answer cannot be found in the context, say "I don't know."
    2. Provide a comprehensive, detailed answer.
    3. At the end of your response, list all unique sources used in the format: "References: [filename1], [filename2]".
    
    Answer:
    """

    response = llm.invoke(prompt)

    return response.content if hasattr(response, 'content') else response



def format_context(qdrant_points, include_tables=True, include_images_as_text=True):
    """
    Tailored for Qdrant payload structure.
    Expects qdrant_points: list of ScoredPoint objects from query_points().
    """
    lines = []
    for i, p in enumerate(qdrant_points, 1):
        payload = p.payload
        # Determine modality from your stored metadata
        m = payload.get("modality", "text") 
        content = payload.get("text", "") # The summary stored in payload
        original = payload.get("original", "")
        filename = payload.get("filename", "Unknown")

        if m == "text":
            lines.append(f"[{i}] TEXT (from {filename}) — Original: {original}")
        elif m == "table" and include_tables:
            lines.append(f"[{i}] TABLE (from {filename}) — Summary: {content}")
        elif m == "image" and include_images_as_text:
            lines.append(f"[{i}] IMAGE (from {filename}) — Summary: {content}")
            
    return "\n\n".join(lines)