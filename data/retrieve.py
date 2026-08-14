from dotenv import load_dotenv
from google import genai
from google.genai import types
import os
import json
import numpy as np
from langchain_core.tools import tool


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

with open("C:/Users/archit/Desktop/ragbot/data/embedded_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

EMBEDDING_MODEL = "gemini-embedding-001"

@tool
def search_portfolio(query: str, top_k: int =3):
    """Use this tool to answer questions about Archit's projects, technical
    skills, education, or certifications. It searches his resume and project
    documentation and returns the most relevant information."""

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
    top_chunks = ranked[:top_k]

    result_text = "\n\n".join(chunk["text"] for score, chunk in top_chunks)
    return result_text


"""# ---- Test with a specific, sharp question ----
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
    print("-" * 40)"""

if __name__ == "__main__":
    print(search_portfolio.invoke({"query": "What was your YOLOv8 model's precision?"}))