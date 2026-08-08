import json
import numpy as np

# ---- Load embedded chunks ----

with open("embedded_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

def cosine_similarity(vec_a, vec_b):
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    return dot_product / (magnitude_a * magnitude_b)

# ---- Pick chunks to compare ----
# Related: two chunks from the same project README (weapon detection)
weapon_chunks = [c for c in chunks if c["source"] == "weapondetectiono_readme.md"]

# Unrelated: one weapon detection chunk vs one from a different project/resume
mentalhealth_chunks = [c for c in chunks if c["source"] == "mentalhealth_readme.md"]
resume_chunks = [c for c in chunks if c["source"] == "resume"]

# ---- Run comparisons ----

print("=== Related pair (both weapon detection) ===")
sim_related = cosine_similarity(weapon_chunks[0]["embedding"], weapon_chunks[1]["embedding"])
print(f"Chunk A: {weapon_chunks[0]['text'][:60]}...")
print(f"Chunk B: {weapon_chunks[1]['text'][:60]}...")
print(f"Cosine similarity: {sim_related:.4f}\n")

print("=== Unrelated pair (weapon detection vs mental health) ===")
sim_unrelated_1 = cosine_similarity(weapon_chunks[0]["embedding"], mentalhealth_chunks[0]["embedding"])
print(f"Chunk A: {weapon_chunks[0]['text'][:60]}...")
print(f"Chunk B: {mentalhealth_chunks[0]['text'][:60]}...")
print(f"Cosine similarity: {sim_unrelated_1:.4f}\n")

print("=== Unrelated pair (weapon detection vs resume/certs) ===")
sim_unrelated_2 = cosine_similarity(weapon_chunks[0]["embedding"], resume_chunks[0]["embedding"])
print(f"Chunk A: {weapon_chunks[0]['text'][:60]}...")
print(f"Chunk B: {resume_chunks[0]['text'][:60]}...")
print(f"Cosine similarity: {sim_unrelated_2:.4f}\n")

print("Sanity check: related similarity should be noticeably higher than unrelated.")