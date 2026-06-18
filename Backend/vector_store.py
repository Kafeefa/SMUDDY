import faiss
import numpy as np
from sentence_transformers import SentenceTransformer  
model= SentenceTransformer("all-MiniLM-L6-v2")
dimension=384
index = faiss.IndexFlatL2(dimension)

chunks_store = []

def add_chunks(chunks):

    embeddings = model.encode(chunks)

    index.add(
        np.array(embeddings).astype("float32")
    )

    chunks_store.extend(chunks)

def search_chunks(query, k=3):

    query_embedding = model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        k
    )

    results = []

    for idx in indices[0]:
        if idx < len(chunks_store):
            results.append(
                chunks_store[idx]
            )

    return results