import faiss
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
dimension = 384
index = faiss.IndexFlatL2(dimension)

chunks_store = []
model = None


def get_model():
    global model

    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME)

    return model

def add_chunks(chunks):
    if not chunks:
        return

    embeddings = get_model().encode(chunks)

    index.add(  
        np.array(embeddings).astype("float32")
    )

    chunks_store.extend(chunks)

def search_chunks(query, k=3):
    if not chunks_store or index.ntotal == 0:
        return []

    query_embedding = get_model().encode([query])

    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        min(k, index.ntotal)
    )

    results = []

    for idx in indices[0]:
        if 0 <= idx < len(chunks_store):
            results.append(
                chunks_store[idx]
            )

    return results
