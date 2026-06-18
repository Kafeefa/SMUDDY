from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

print("Loading model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "Java is a programming language",
    "Python is used in AI",
    "Football is a sport"
]

embeddings = model.encode(texts)

print("Embedding shape:", embeddings.shape)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings).astype("float32"))

print("Vectors stored:", index.ntotal)

query = model.encode(["AI and machine learning"])

distances, indices = index.search(
    np.array(query).astype("float32"),
    k=2
)

print("Indices:", indices)
print("Distances:", distances)

for idx in indices[0]:
    print(texts[idx])