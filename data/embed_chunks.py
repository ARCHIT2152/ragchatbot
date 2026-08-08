import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

EMBEDDING_MODEL = "gemini-embedding-001"

with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)


for i, chunk in enumerate(chunks):
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=chunk["text"],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )

    chunk["embedding"] = result.embeddings[0].values

    print(f"Embedded chunk {i+1}/{len(chunks)}")
    time.sleep(0.1)

with open("embedded_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print(f"Done. Saved {len(chunks)} embedded chunks to embedded_chunks.json")