import faiss
import numpy as np
import pickle
import os

INDEX_PATH = "faiss_index.bin"
META_PATH  = "chunks_meta.pkl"

def build_index(embeddings: list, chunks: list):
    """
    Builds a FAISS index from embeddings and saves metadata.
    """

    # ✅ Safety check added
    if len(embeddings) == 0:
        raise ValueError("No embeddings created. PDF may not contain extractable text.")

    dim = len(embeddings[0])
    vectors = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # Save index
    faiss.write_index(index, INDEX_PATH)

    # Save metadata
    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks


def load_index():
    """
    Load saved FAISS index + metadata
    """
    if not os.path.exists(INDEX_PATH):
        return None, None

    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def search(query_embedding: list, index, chunks, top_k=5):
    """
    Search top_k similar chunks
    """
    query_vec = np.array([query_embedding], dtype="float32")
    distances, indices = index.search(query_vec, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(chunks):
            results.append(chunks[idx])

    return results