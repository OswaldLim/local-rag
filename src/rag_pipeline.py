from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_qdrant import QdrantVectorStore
import uuid
from datetime import datetime

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://ollama:11434"
)

llm = OllamaLLM(
    model="llama3.2",
    base_url="http://ollama:11434"
)

qdrant = QdrantClient(host="qdrant", port=6333)
COLLECTION_NAME = "rag_docs"

qdrant.delete_collection(collection_name=COLLECTION_NAME)

if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": 768, "distance": "Cosine"},
    )

vector_store = QdrantVectorStore(
    client=qdrant,
    collection_name=COLLECTION_NAME,
    embedding=embedding_model
)

def ingest_document(doc_name: str, text_chunks: list, type: str = "text"):
    points = []
    print(f"INGESTING DOCUMENTSSSS\n  {text_chunks}", flush=True)
    for i, chunk in enumerate(text_chunks):
        vector = embedding_model.embed_query(chunk)
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text":chunk, "doc_name":doc_name, "timestamps":datetime.now(), "type":type}
        ))
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)

def query_rag(query: str, top_k: int = 6) -> str:
    # Embed query
    query_vector = embedding_model.embed_query(query)
    result = qdrant.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=top_k)
    # result = vector_store.similarity_search_by_vector(
    #     embedding=query_vector,
    #     k=top_k
    # )
    print("FINISH QUERYing points!!!!!")

    list_of_scored_points = [tups for scored_points in result for tups in scored_points][1]
    context = "\n\n".join([f"text: {text.payload["text"]}, file: {text.payload["doc_name"]}" for text in list_of_scored_points])


    prompt = f"Answer the following question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query}\nshow the doc_name at the end of the answer as reference - there can be multiple doc_names - only if you know else say you don't know if no context is provided\nAnswer:"

    response = llm.invoke(prompt)

    return response

