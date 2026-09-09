import os
import pickle
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PAPERS_DIR = "data/papers"
VECTOR_DB_DIR = "vector_db"
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 150         # overlap between consecutive chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"        # small, fast


def load_pdf_text(filepath: str) -> str:
    """Extract all text from a PDF file"""
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0 
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def main():
    if not os.path.isdir(PAPERS_DIR) or not os.listdir(PAPERS_DIR):
        print(f"No PDFs found in {PAPERS_DIR}/. Add your papers there first.")
        return
    
    os.makedirs(VECTOR_DB_DIR, exist_ok = True)

    print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    all_chunks = []     # list of chunk text
    all_metadate = []       # parallel list of dicts: {paper, chunk_index}
    # all_ids = []    # Chroma requires a unique string id per entry

    pdf_files = [f for f in os.listdir(PAPERS_DIR) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")

    for filename in pdf_files:
        paper_name = os.path.splitext(filename)[0]
        filepath = os.path.join(PAPERS_DIR, filename)
        print(f"  Reading {filename} ...")
        text = load_pdf_text(filepath)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f" -> {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadate.append({"paper": paper_name, "chunk_index": i})
            # all_ids.append(f"{paper_name}_{i}")

    print(f"\nEmbedding {len(all_chunks)} chunks total ...")
    embeddings = model.encode(all_chunks, show_progress_bar = True, convert_to_numpy= True)
    embeddings = embeddings.astype("float32")

    # print("Writing to ChromaDB ...")
    # client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    # try:
        # client.delete_collection("papers")
    # except Exception:
    #    pass
    # collection = client.create_collection("papers")

    # Normalize so inner-product search behaves like cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)    # IP = inner product (cosine, since normalized)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(VECTOR_DB_DIR, "faiss.index"))
    with open(os.path.join(VECTOR_DB_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump({"chunks": all_chunks, "metadata": all_metadate}, f)

    # collection.add(
    #       documents = all_chunks, embeddings = embeddings.tolist(), metadatas=all_metadata, ids=all_ids)
    # )

    print(f"\nDone. Indexed {len(all_chunks)} chunks from {len(pdf_files)} papers.")
    print(f"Saved to {VECTOR_DB_DIR}/faiss.index and {VECTOR_DB_DIR}/metadata.pkl")


if __name__ == "__main__":
    main()