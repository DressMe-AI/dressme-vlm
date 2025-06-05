import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Load semantic descriptions
SEMANTICS_PATH = Path("output/semantics.json")
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

