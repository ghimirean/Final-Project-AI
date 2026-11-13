import re
import json
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

def find(pattern, text):
    match = re.search(pattern, text)
    return match.group() if match else None

def extract_contact(header_lines):
    header = " ".join(header_lines)
    return {
        "name": header_lines[0] if header_lines else None,
        "email": find(r"[\w.+-]+@[\w-]+\.[\w.-]+", header),
        "phone": find(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}", header),
        "linkedin": find(r"linkedin\.com/in/[\w-]+", header),
        "github": find(r"github\.com/[\w-]+", header),
    }

if __name__ == "__main__":
    text = read_pdf("data/sample.pdf")
    sections = split_sections(text)
    contact = extract_contact(sections["HEADER"])
    print(json.dumps(contact, indent=2))