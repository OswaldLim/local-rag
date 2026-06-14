from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import CrossEncoder
import uuid
import asyncio

import time
import logging

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://ollama:11434"
)

llm = OllamaLLM(
    model="llama3.2:3b",
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

async def ingest_with_buffer(document_generator, buffer_size=10):
    buffer = []
    count = 0
    
    # Iterate through the async generator
    async for doc in document_generator:
        buffer.append(doc)
        
        # When buffer hits the limit, ingest and clear
        if len(buffer) >= buffer_size:
            await ingest_batch(buffer)
            count += len(buffer)
            buffer = [] # Clear buffer
            
    # Ingest any remaining items in the buffer
    if buffer:
        await ingest_batch(buffer)
        count += len(buffer)
        
    yield count

async def ingest_batch(text_chunks: list):
    if not text_chunks:
        return 0
    print(f"INGESTING DOCUMENTSSSS\n  {text_chunks}", flush=True)
    
    texts = [chunk.page_content for chunk in text_chunks]

    vectors = await embedding_model.aembed_documents(texts)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[i],
            payload={
                "text": chunk.page_content,
                **chunk.metadata
            }
        )
        for i, chunk in enumerate(text_chunks)
    ]
    
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


async def get_rag_context_async(query: str, top_k: int = 20):
    # Use loop.run_in_executor for CPU-bound tasks like embedding and reranking
    loop = asyncio.get_running_loop()
    
    # Run synchronous operations in a thread pool so they don't block the event loop
    query_vector = await loop.run_in_executor(None, embedding_model.embed_query, query)
    
    # If your qdrant client is the async version, use await qdrant.aquery_points(...)
    result = qdrant.query_points(
        collection_name=COLLECTION_NAME, 
        query=query_vector,
        limit=top_k
    )
    
    optimized_result = await loop.run_in_executor(None, rerank_results, query, result.points, 5)
    return format_context(optimized_result)

# def build_prompt(query: str, top_k: int = 20) -> str:
#     start = time.perf_counter()
#     # Embed query
#     query_vector = embedding_model.embed_query(query)
#     result = qdrant.query_points(
#             collection_name=COLLECTION_NAME, 
#             query=query_vector,
#             limit=top_k)
#     print("FINISH QUERYing points!!!!!")
#     print(f"Retrieval took: {time.perf_counter() - start:.2f}s")

#     optimized_result = rerank_results(query, result.points, top_n=5)
#     print(f"Reranking took: {time.perf_counter() - start:.2f}s")

#     # New version
#     context = format_context(optimized_result)

#     # print(f"ContextSTTTTT\n\n{context}\n\n")

#     prompt = f"""
#     You are an expert assistant. Answer the question using ONLY the context provided below.
    
#     Context:
#     {context}
    
#     Question: {query}
    
#     Instructions:
#     1. If the answer cannot be found in the context, say "I don't know."
#     2. Provide a comprehensive and concise answer.
#     3. At the end of your response, list all file names used in the final answer: "References: [filename1], [filename2]".
    
#     Answer:
#     """

#     response = llm.invoke(prompt)
#     print(f"LLM Generation took: {time.perf_counter() - start:.2f}s")

#     return response.content if hasattr(response, 'content') else response

async def stream_rag_response(prompt):
    start_time = time.perf_counter()

    context = await get_rag_context_async(prompt)

    prompt = f"""
        You are an expert assistant. Answer the question using ONLY the context provided below.
        
        Context:
        {context}
        
        Question: {prompt}
        
        Instructions:
        1. If the answer cannot be found in the context, say "I don't know."
        2. Provide a comprehensive and concise answer.
        3. At the end of your response, list all file names used in the final answer: "References: [filename1], [filename2]".
        
        Answer:
    """
    
    first_token_logged = False
    # .stream() returns an async generator
    async for chunk in llm.astream(prompt):
        if not first_token_logged:
            ttft = time.perf_counter() - start_time
            logging.info(f"Time to First Token (TTFT): {ttft:.2f}s")
            first_token_logged = True
        # Access the content depending on the model's structure
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        yield content

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

def rerank_results(query, points, top_n=5):
    # 1. Prepare pairs for the model
    pairs = [[query, p.payload.get("original", "")] for p in points]
    
    # 2. Get scores
    scores = reranker.predict(pairs)
    
    # 3. Combine with original points and sort
    scored_results = sorted(zip(points, scores), key=lambda x: x[1], reverse=True)
    
    # 4. Return top N points
    return [item[0] for item in scored_results[:top_n]]
