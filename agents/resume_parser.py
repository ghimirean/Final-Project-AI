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

def load_skills(path="data/skills.txt"):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

def extract_skills(text, skill_list):
    found = []
    for skill in skill_list:
        # Match the skill as a whole word, so "Java" doesn't match "JavaScript"
        pattern = r"(?<![\w+#])" + re.escape(skill) + r"(?![\w+#])"
        # Short skills like "C" or "R" must match exact case
        flags = 0 if len(skill) <= 2 else re.IGNORECASE
        if re.search(pattern, text, flags):
            found.append(skill)
    return found

if __name__ == "__main__":
    text = read_pdf("data/sample.pdf")
    sections = split_sections(text)
    profile = extract_contact(sections["HEADER"])
    profile["skills"] = extract_skills(text, load_skills())
    print(json.dumps(profile, indent=2))