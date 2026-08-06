input_path = "data/resume_extracted.txt"
output_path = "data/resume_marked.txt"

with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

marked_lines = []

for line in lines:
    stripped = line.strip()

    if stripped and stripped.isupper():
        marked_lines.append(f"##HEADER##{stripped}\n")
    else:
        marked_lines.append(line)

marked_text = "".join(marked_lines)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(marked_text)

print(f"Processed {len(lines)} lines, wrote marked output to {output_path}")