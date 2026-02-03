from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_qdrant import QdrantVectorStore
import uuid

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
        vectors_config={"size": 768, "distance": "Cosine"}
    )

vector_store = QdrantVectorStore(
    client=qdrant,
    collection_name=COLLECTION_NAME,
    embedding=embedding_model
)

def ingest_document(doc_name: str, text_chunks: list):
    points = []
    for i, chunk in enumerate(text_chunks):
        vector = embedding_model.embed_query(chunk)
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text":chunk, "doc_name":doc_name}
        ))
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)

def query_rag(query: str, top_k: int = 5) -> str:
    # Embed query
    query_vector = embedding_model.embed_query(query)

    result = qdrant.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=top_k)
    # result = vector_store.similarity_search_by_vector(
    #     embedding=query_vector,
    #     k=top_k
    # )


    list_of_scored_points = [tups for scored_points in result for tups in scored_points][1]
    context = "\n\n".join([text.payload["text"] for text in list_of_scored_points])


    prompt = f"Answer the following question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    response = llm.invoke(prompt)

    return response


# import utils

# if __name__ == "__main__":
#     file = "C:\\Users\\Lenovo\\Desktop\\SCHOOL\\Personal\\Internship\\local_rag_docker\\temp_3 Logic (3A) [student].pdf"

#     file2 = "C:\\Users\\Lenovo\\Desktop\\SCHOOL\\Year 1\\Math\\Lecture Slides\\1 Mathematical Proof (1A)(v2) [Student].pdf"

#     pages = utils.load_pdf(file)
#     all_chunks = []
#     for page in pages:
#         all_chunks.extend(utils.chunk_text(page))
    
#     count = ingest_document(file, all_chunks)
#     dictionary = {"message": f"Ingested {count} chunks from {file}"}
#     print(dictionary)

#     pages = utils.load_pdf(file2)
#     all_chunks = []
#     for page in pages:
#         all_chunks.extend(utils.chunk_text(page))
#     count = ingest_document(file2, all_chunks)
#     dictionary = {"message": f"Ingested {count} chunks from {file2}"}
#     print(dictionary)


#     answer = query_rag("What is propositional logic")
#     print({"answer": answer})