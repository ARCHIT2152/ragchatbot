from dotenv import load_dotenv
from google import genai
from google.genai import types
import os
import json
import numpy as np

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def cosine_similarity(vec_a, vec_b):
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    return dot_product / (magnitude_a * magnitude_b)

with open("embedded_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

EMBEDDING_MODEL = "gemini-embedding-001"

def search_portfolio(query, top_k=3):
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    query_embedding = result.embeddings[0].values

    scores = []
    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scores.append((score, chunk))

    ranked = sorted(scores, key=lambda x: x[0], reverse=True)

    return ranked[:top_k]


# ---- Test with a specific, sharp question ----
print("=== Query: 'What was your YOLOv8 model's precision?' ===")
results = search_portfolio("What was your YOLOv8 model's precision?")
for score, chunk in results:
    print(f"Score: {score:.4f} | Source: {chunk['source']}")
    print(f"Text: {chunk['text'][:200]}...")
    print("-" * 40)

# ---- Test with a broader, vaguer question ----
print("\n=== Query: 'What model did you use for weapon detection, and what precision did it achieve??' ===")
results = search_portfolio("What model did you use for weapon detection, and what precision did it achieve?")
for score, chunk in results:
    print(f"Score: {score:.4f} | Source: {chunk['source']}")
    print(f"Text: {chunk['text'][:200]}...")
    print("-" * 40)