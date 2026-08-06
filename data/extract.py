import pdfplumber
pdf_path = "data/archit_resume.pdf"
output_path = "data/resume_extracted.txt"
with pdfplumber.open(pdf_path) as pdf:
    all_text = []
    for i, page in enumerate(pdf.pages):
        text = page.extract_text(layout=True)
        all_text.append(f"--- Page {i+1} ---\n{text}\n")

full_text = "\n".join(all_text)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"Extracted {len(pdf.pages)} page(s) to {output_path}")