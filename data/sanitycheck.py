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
# Get all weapon detection chunks first, then filter for substantive/technical ones
weapon_chunks = [c for c in chunks if c["source"] == "weapondetectiono_readme.md"]
mentalhealth_chunks = [c for c in chunks if c["source"] == "mentalhealth_readme.md"]

# Related: two DIFFERENT technical chunks from weapon detection (not title/ToC)
weapon_technical_1 = [c for c in weapon_chunks if "knife" in c["text"] or "pistol" in c["text"]]
weapon_technical_2 = [c for c in weapon_chunks if "surveillance" in c["text"] or "alert" in c["text"] or "confidence threshold" in c["text"]]

# Unrelated: substantive weapon detection chunk vs substantive mental health chunk
mentalhealth_technical = [c for c in mentalhealth_chunks if "XGBoost" in c["text"] or "accuracy" in c["text"]]

# ---- Run comparisons ----

print("=== Related pair (both weapon detection, technical content) ===")
sim_related = cosine_similarity(weapon_technical_1[0]["embedding"], weapon_technical_2[0]["embedding"])
print(f"Chunk A: {weapon_technical_1[0]['text'][:60]}...")
print(f"Chunk B: {weapon_technical_2[0]['text'][:60]}...")
print(f"Cosine similarity: {sim_related:.4f}\n")

print("=== Unrelated pair (weapon detection vs mental health, technical content) ===")
sim_unrelated_1 = cosine_similarity(weapon_technical_1[0]["embedding"], mentalhealth_technical[0]["embedding"])
print(f"Chunk A: {weapon_technical_1[0]['text'][:60]}...")
print(f"Chunk B: {mentalhealth_technical[0]['text'][:60]}...")
print(f"Cosine similarity: {sim_unrelated_1:.4f}\n")


print("Sanity check: related similarity should be noticeably higher than unrelated.")