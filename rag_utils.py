import os 
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

VECTOR_DB_DIR = "vector_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-sonnet-4-5"
TOP_K = 5 

_embedding_model = None
_faiss_index = None
_chunks = None
_metadata = None
_anthropic_client = None


def _load_resources():
    """Load the model, FAISS index and metadata once"""
    global _embedding_model, _faiss_index, _chunks, _metadata, _anthropic_client

    # global _embedding_model, _collection, _anthropic_client

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    
    if _faiss_index is None:
        index_path = os.path.join(VECTOR_DB_DIR, "faiss.index")
        if not os.path.exists(index_path):
            raise FileExistsError(f"{index_path} not found. Run `python ingest.py` first.")
        _faiss_index = faiss.read_index(index_path)

    # if _collection is None:
    #     if not os.path.exists(VECTOR_DB_DIR):
    #         raise FileNotFoundError(
    #             f"{VECTOR_DB_DIR}/ not found. Run `python ingest.py` first.")

    # No separate metadata.pkl to load -- Chroma keeps text + metadata together with the vectors.
    # client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    # _collection = client.get_collection("papers")

    if _chunks is None or _metadata is None:
        metapath = os.path.join(VECTOR_DB_DIR, "metadata.pkl")
        with open(metapath, "rb") as f:
            data = pickle.load(f)
        _chunks = data["chunks"]
        _metadata = data["metadata"]

    if _anthropic_client is None:
        _anthropic_client = Anthropic()


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed the query and search FAISS for the most similar chunks.
    Returns a list of dicts: {text, paper, chunk_index, score}
    """
    _load_resources()

    """
    Embed the query and search Chroma for the most similar chunks.
    Returns a list of dicts: {text, paper, chunk_index, score}
    -- SAME return shape as the FAISS version, so nothing downstream
    (answer_question, server.py's tools, app.py) needs to change.
    """
    # _load_resources()

    # query_vector = _embedding_model.encode([query], convert_to_numpy=True).tolist()

    # results = _collection.query(query_embeddings=query_vector, n_results=top_k)

    # output = []
    # documents = results["documents"][0]
    # metadatas = results["metadatas"][0]
    # distances = results["distances"][0]

    # for text, meta, distance in zip(documents, metadatas, distances):
    #     output.append({
    #         "text": text,
    #         "paper": meta["paper"],
    #         "chunk_index": meta["chunk_index"],
    #         # Chroma returns a distance (lower = more similar); convert to a
    #         # similarity-style score so it still reads the same as before
    #         "score": round(1 - distance, 3),
    #     })
    # return output

    # def list_indexed_papers() -> list[str]:
    #     _load_resources()
    #     all_data = _collection.get()  # fetches all stored metadata
    #     papers = sorted(set(m["paper"] for m in all_data["metadatas"]))
    #     return papers

    query_vector = _embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vector)

    scores, indices = _faiss_index.search(query_vector, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "text": _chunks[idx],
            "paper": _metadata[idx]["paper"],
            "chunk_index": _metadata[idx]["chunk_index"],
            "score": float(score)
        })
    return results

def _build_prompt(query: str, chunks: list[dict]) -> str:
    """Combine retrieved chunks into a single prompt for Claude."""
    context_blocks = []
    for c in chunks:
        context_blocks.append(
            f"[Source: {c['paper']}, chunk {c['chunk_index']}]\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    return(
        "You are a research assistant answering questions using only the provided "
        "paper excerpts. Cite the paper name for every claim you make. "
        "If the excerpts don't contain the answer, say so honestly.\n\n"
        f"EXCERPTS:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER (with citations to paper names):"
    )


def answer_question(query: str, top_k: int = TOP_K) -> dict:
    """
    Full RAG pipeline: retrieve relevant chunks, generate an answer, return both.
    Returns: {"answer": str, "sources": list[dict]}
    """
    _load_resources()
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {"answer": "No relevant content found in the indexed papers.", "sources": []}
    
    prompt = _build_prompt(query, chunks)

    response = _anthropic_client.messages.create(
        model= CLAUDE_MODEL,
        max_tokens= 1000,
        messages= [{"role": "user", "content": prompt}]
    )
    answer_text = "".join(
        block.text for block in response.content if block.type == "text"
    )

    # Deduplicate sources (same paper may appear via multiple chunks)
    seen = set()
    sources = []
    for c in chunks:
        if c["paper"] not in seen:
            sources.append({"paper": c["paper"], "score": round(c["score"], 3)})
            seen.add(c["paper"])
    
    return {"answer": answer_text, "sources": sources}