import pdfplumber

def read_pdf(path):
       text = ""
       with pdfplumber.open(path) as pdf:
           for page in pdf.pages:
               text += (page.extract_text() or "") + "\n"
       return text

SECTION_KEYWORDS = ["EDUCATION", "EXPERIENCE", "SKILLS", "PROJECTS",
                    "CERTIFICATIONS", "LEADERSHIP", "ACTIVITIES",
                    "SUMMARY", "OBJECTIVE", "AWARDS", "LANGUAGES"]

def split_sections(text):
    sections = {"HEADER": []}
    current = "HEADER"
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # A heading is an ALL-CAPS line containing a known keyword
        if line.isupper() and any(k in line for k in SECTION_KEYWORDS):
            current = line
            sections[current] = []
        else:
            sections[current].append(line)
    return sections

if __name__ == "__main__":
    text = read_pdf("data/sample.pdf")
    sections = split_sections(text)
    for name, lines in sections.items():
        print(f"===== {name} ({len(lines)} lines) =====")
        for l in lines:
            print("   ", l)