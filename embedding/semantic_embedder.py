import os
import json
import numpy as np
import faiss
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Load semantic descriptions
SEMANTICS_PATH = Path("../output/semantics.json")
with open(SEMANTICS_PATH, "r") as f:
    items = json.load(f)

# Embed all descriptions
descriptions = [item["description"] for item in items]
ids = [item["id"] for item in items]

response = client.embeddings.create(
    input=descriptions,
    model="text-embedding-3-small"
)

embeddings = [e.embedding for e in response.data]
print(f"Loaded and embedded {len(embeddings)} descriptions.")

# Normalize embeddings
embedding_dim = len(embeddings[0])
embedding_matrix = np.array(embeddings).astype("float32")
faiss.normalize_L2(embedding_matrix)

# Build FAISS index
index = faiss.IndexFlatIP(embedding_dim)  # Cosine similarity via dot product on L2-normalized vectors
index.add(embedding_matrix)

# Save index
faiss.write_index(index, "../output/semantic_index.faiss")

# Save ID mapping
with open("../output/id_map.json", "w") as f:
    json.dump(ids, f)

print(f"FAISS index built and saved with {index.ntotal} items.")