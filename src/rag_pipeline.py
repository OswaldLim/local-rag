from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_qdrant import QdrantVectorStore
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
import uuid
from datetime import datetime

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://ollama:11434"
)

bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")
late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
dense_embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

llm = OllamaLLM(
    model="llama3.2",
    base_url="http://ollama:11434"
)

qdrant = QdrantClient(host="qdrant", port=6333)
COLLECTION_NAME = "rag_docs"

#remember to remove for final work
if qdrant.collection_exists(COLLECTION_NAME):
    qdrant.delete_collection(collection_name=COLLECTION_NAME)

def checkDatabaseExist(dense_embeddings, late_interaction_embeddings):
    if not qdrant.collection_exists(COLLECTION_NAME):
        print("CREATE COLLECTION")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "all-MiniLM-L6-v2": models.VectorParams(
                    size=len(dense_embeddings[0]),
                    distance=models.Distance.COSINE,
                ),
                "colbertv2.0": models.VectorParams(
                    size=len(late_interaction_embeddings[0][0]),
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM,
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0)  #  Disable HNSW for reranking
                ),
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF
                )
            }
        )

# vector_store = QdrantVectorStore(
#     client=qdrant,
#     collection_name=COLLECTION_NAME,
#     embedding=embedding_model
# )

def ingest_document(doc_name: str, text_chunks: list, type: str = "text"):
    
    dense_embeddings = list(dense_embedding_model.embed(text_chunks))
    bm25_embeddings = list(bm25_embedding_model.embed(text_chunks))
    late_interaction_embeddings = list(late_interaction_embedding_model.embed(text_chunks))
    checkDatabaseExist(dense_embeddings, late_interaction_embeddings)

    points = []
    print(f"INGESTING DOCUMENTSSSS\n  {text_chunks}", flush=True)
    if len(text_chunks) == 0:
        return 0
    for idx, (dense_embedding, bm25_embedding, late_interaction_embedding, doc) in enumerate(zip(dense_embeddings, bm25_embeddings, late_interaction_embeddings, text_chunks)):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "all-MiniLM-L6-v2": dense_embedding,
                "bm25": bm25_embedding.as_object(),
                "colbertv2.0": late_interaction_embedding,
            },
            payload={"text": doc, "doc_name": doc_name, "doc_type": type}
        )
        points.append(point)

    operation_info = qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(f"OPERATION INFO {operation_info}")

    return len(points)

def query_rag(query: str, top_k: int = 7) -> str:
    # Embed query
    dense_vectors = next(dense_embedding_model.query_embed(query))
    sparse_vectors = next(bm25_embedding_model.query_embed(query))
    late_vectors = next(late_interaction_embedding_model.query_embed(query))
    
    prefetch = [
        models.Prefetch(
            query=dense_vectors,
            using="all-MiniLM-L6-v2",
            limit=20,
        ),
        models.Prefetch(
            query=models.SparseVector(**sparse_vectors.as_object()),
            using="bm25",
            limit=20,
        ),
    ]

    result = qdrant.query_points(
            COLLECTION_NAME,
            prefetch=prefetch,
            query=late_vectors,
            using="colbertv2.0",
            with_payload=True,
            limit=10,
    )
    
    # query_vector = embedding_model.embed_query(query)
    # if "table" in query:
    #     query_filter = models.Filter(
    #         must=[
    #             models.FieldCondition(
    #                 key="chunk_type",
    #                 match=models.MatchValue(value="tables")
    #             ),
    #             models.FieldCondition(
    #                 key="text",
    #                 match=models.MatchTextAny(text_any=query),
    #             )
    #         ]
    #     )
    #     result = qdrant.query_points(
    #         collection_name=COLLECTION_NAME, 
    #         query=query_vector,
    #         with_payload=True,
    #         query_filter=query_filter,
    #         limit=top_k)
    # else:
    #     result = qdrant.query_points(
    #         collection_name=COLLECTION_NAME, 
    #         query=query_vector,
    #         limit=top_k)
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

