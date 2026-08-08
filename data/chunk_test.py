from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import json


# ---- Step 1: Load and prepare the resume (excluding PROJECTS section) ----

with open("resume_marked.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()

resume_text = re.sub(r"--- Page \d+ ---\n?", "", resume_text)

# Split resume into sections using our ##HEADER## marker
resume_sections = resume_text.split("##HEADER##")


# Drop the section that starts with "PROJECTS" — we're sourcing that from READMEs instead
resume_sections_filtered = [
    section for section in resume_sections
    if not section.strip().upper().startswith("P R O J E C T S")
]

resume_text_filtered = "##HEADER##".join(resume_sections_filtered)

# ---- Step 2: Load the 3 README files ----

readme_files = [
    "energy_readme.md",
    "mentalhealth_readme.md",
    "weapondetectiono_readme.md",
]

readme_texts = []
for path in readme_files:
    with open(path, "r", encoding="utf-8") as f:
        readme_texts.append(f.read())

# ---- Step 3: Set up the splitter ----

splitter = RecursiveCharacterTextSplitter(
    separators=["##HEADER##", "\n## ", "\n\n", "\n", " "],
    chunk_size=700,
    chunk_overlap=0,
)

# ---- Step 4: Split each source, tagging chunks with their origin ----

all_chunks = []

resume_chunks = splitter.split_text(resume_text_filtered)
for chunk in resume_chunks:
    all_chunks.append({"source": "resume", "text": chunk})

for path, text in zip(readme_files, readme_texts):
    readme_chunks = splitter.split_text(text)
    for chunk in readme_chunks:
        all_chunks.append({"source": path, "text": chunk})

# ---- Step 5: Save chunks to inspect ----

with open("chunks_output.txt", "w", encoding="utf-8") as f:
    for i, item in enumerate(all_chunks):
        f.write(f"--- Chunk {i+1} | source: {item['source']} ---\n")
        f.write(item["text"])
        f.write("\n\n")

print(f"Total chunks created: {len(all_chunks)}")
print(f"Saved to chunks_output.txt for inspection")

with open("chunks.json", "w",encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2)