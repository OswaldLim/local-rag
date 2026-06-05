from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_qdrant import QdrantVectorStore
from fastembed import SparseTextEmbedding, LateInteractionTextEmbedding
import uuid

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://ollama:11434"
)

sparse_embedding_model = SparseTextEmbedding("Qdrant/bm25")
late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")


llm = OllamaLLM(
    model="llama3.2",
    base_url="http://ollama:11434"
)

qdrant = QdrantClient(host="qdrant", port=6333)
COLLECTION_NAME = "rag_docs"

qdrant.delete_collection(collection_name=COLLECTION_NAME)

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": 768, "distance": "Cosine"},
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
        vector = embedding_model.embed_query(chunk)
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=chunk.metadata
        ))
        sparse_points.append
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)

def query_rag(query: str, top_k: int = 7) -> str:
    # Embed query
    query_vector = embedding_model.embed_query(query)
    if "table" in query:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="chunk_type",
                    match=models.MatchValue(value="tables")
                ),
                models.FieldCondition(
                    key="text",
                    match=models.MatchTextAny(text_any=query),
                )
            ]
        )
        result = qdrant.query_points(
            collection_name=COLLECTION_NAME, 
            query=query_vector,
            with_payload=True,
            query_filter=query_filter,
            limit=top_k)
    else:
        result = qdrant.query_points(
            collection_name=COLLECTION_NAME, 
            query=query_vector,
            limit=top_k)
    print("FINISH QUERYing points!!!!!")


    list_of_scored_points = [tups for scored_points in result for tups in scored_points][1]

    print(f"\n\n{list_of_scored_points}\n\n")

    context = "\n\n".join([f"text: {text.payload["text"]}, file: {text.payload["doc_name"]}" for text in list_of_scored_points])


    print(f"ContextSTTTTT\n\n{context}\n\n")

    prompt = f"""
    You are an assistant answering based on provided context.

    Instructions:
    - Use ALL relevant information from the context
    - Provide a COMPLETE answer
    - Do not omit important details
    - If the answer spans multiple parts, include all of them
    - Answer only if the context is relevant else say you don't know
    - Answer don't know if no context is provided

    Context:
    {context}
    
    Question: {query}
    
    show the doc_name at the end of the answer as reference - there can be multiple doc_names in the form of doc_name: document name.
    
    Answer:
    """

    response = llm.invoke(prompt)

    return response

