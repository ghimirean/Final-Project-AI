from datetime import date
import re
import json
import os
import pdfplumber
import docx


def read_pdf(path):
       text = ""
       with pdfplumber.open(path) as pdf:
           for page in pdf.pages:
               text += (page.extract_text() or "") + "\n"
       return text

def read_docx(path):
    document = docx.Document(path)
    lines = [p.text for p in document.paragraphs]
    # Some resume templates put content inside tables, so read those cells too
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                lines.append(cell.text)
    return "\n".join(lines)

def read_resume(path):
    # Picks the right reader based on the file extension
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    raise ValueError(f"Unsupported file type: {ext} (use .pdf or .docx)")

SECTION_KEYWORDS = ["EDUCATION", "EXPERIENCE", "SKILLS", "PROJECTS",
                    "CERTIFICATIONS", "LEADERSHIP", "ACTIVITIES",
                    "SUMMARY", "OBJECTIVE", "AWARDS", "LANGUAGES"]

SMALL_WORDS = {"and", "of", "&", "/", "-"}

def is_heading(line):
    # A heading is a short line containing a known keyword, written either
    # in ALL CAPS ("WORK EXPERIENCE") or Title Case ("Work Experience")
    line = line.rstrip(":").strip()
    words = line.split()
    if not words or len(words) > 4:
        return False
    if not any(k in line.upper() for k in SECTION_KEYWORDS):
        return False
    return line.isupper() or all(w[0].isupper() or w.lower() in SMALL_WORDS
                                 for w in words)

def split_sections(text):
    sections = {"HEADER": []}
    current = "HEADER"
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if is_heading(line):
            # Store names in upper case so get_section("EXPERIENCE") always works
            current = line.rstrip(":").strip().upper()
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

# Matches links like https://..., www..., linkedin.com/in/x, github.com/x,
# plus a one-word label right before them ("GitHub: github.com/x").
# Only common web endings are listed, so "Node.js" is NOT treated as a link.
URL_PATTERN = (r"(?:\b[A-Za-z]+:\s*)?"
               r"(?:https?://\S+|www\.\S+"
               r"|\b[\w.-]+\.(?:com|org|net|io|dev|edu|me|co|ai|app)\b(?:/\S*)?)")

def remove_urls(text):
    return re.sub(URL_PATTERN, " ", text, flags=re.IGNORECASE)

def extract_skills(text, skill_list):
    # Drop links first so "github.com/me" doesn't count as the skill "GitHub"
    text = remove_urls(text)
    found = []
    for skill in skill_list:
        # Match the skill as a whole word, so "Java" doesn't match "JavaScript"
        pattern = r"(?<![\w+#])" + re.escape(skill) + r"(?![\w+#])"
        # Short skills like "C" or "R" must match exact case
        flags = 0 if len(skill) <= 2 else re.IGNORECASE
        if re.search(pattern, text, flags):
            found.append(skill)
    return found


def get_section(sections, keyword):
    # Finds a section by keyword, e.g. "EDUCATION" matches "EDUCATION & TRAINING"
    for name, lines in sections.items():
        if keyword in name:
            return lines
    return []

def extract_education(lines):
    text = "\n".join(lines)
    institution = next((l for l in lines
                        if re.search(r"University|College|Institute|School", l)), None)
    degree = next((l for l in lines
                   if re.search(r"\b(Bachelor|Master|Associate|Doctor|Ph\.?D|B\.S|M\.S|B\.A|M\.A|MBA)\b", l)), None)
    field = degree.split(" in ", 1)[1].strip() if degree and " in " in degree else None
    graduation = find(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{4}", text)
    return {
        "institution": institution,
        "degree": degree,
        "field": field,
        "graduation": graduation,
    }

MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
          "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
# One date can be "Aug 2024", "08/2024", or just "2024"
DATE = (r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}"
        r"|\d{1,2}/\d{4}|\d{4})")
DATE_RANGE = (r"\b(" + DATE + r")\s*(?:[–—-]|to)\s*(" + DATE
              + r"|Present|Current|Now)\b")

def to_months(token, is_end=False):
    # Turns a date token into a month count so two dates can be subtracted
    token = token.strip()
    if token.lower() in ("present", "current", "now"):
        return date.today().year * 12 + date.today().month
    if "/" in token:                                  # "08/2024"
        month, year = token.split("/")
        return int(year) * 12 + int(month)
    parts = token.split()
    if len(parts) == 2:                               # "Aug 2024"
        return int(parts[1]) * 12 + MONTHS[parts[0][:3].lower()]
    # "2024" alone: a start year means January, an end year means December
    return int(token) * 12 + (12 if is_end else 1)

def extract_experience(lines):
    jobs = []
    for i, line in enumerate(lines):
        m = re.search(DATE_RANGE, line, re.IGNORECASE)
        if not m:
            continue
        start = to_months(m.group(1))
        end = to_months(m.group(2), is_end=True)
        # The line above the date line is usually "Company | Title"
        header = lines[i - 1] if i > 0 else ""
        if "|" in header:
            company, title = [p.strip() for p in header.split("|", 1)]
        else:
            company, title = header, None
        jobs.append({"company": company, "title": title,
                     # +1 counts both the first and last month (Jan-Dec = 12)
                     "dates": m.group(0), "months": end - start + 1})
    total_years = round(sum(j["months"] for j in jobs) / 12, 1)
    return jobs, total_years


def parse_resume(path, skills_path="data/skills.txt"):
    # Runs the full pipeline on one resume and returns the profile dict
    text = read_resume(path)
    sections = split_sections(text)
    profile = extract_contact(sections.get("HEADER", []))
    profile["skills"] = extract_skills(text, load_skills(skills_path))
    profile["education"] = extract_education(get_section(sections, "EDUCATION"))
    jobs, years = extract_experience(get_section(sections, "EXPERIENCE"))
    profile["experience"] = jobs
    profile["experience_years"] = years
    return profile

def save_profile(profile, out_path="output/profile.json"):
    # Creates the output folder if needed; ensure_ascii=False keeps
    # characters like "é" readable instead of "é"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.pdf"
    profile = parse_resume(path)
    save_profile(profile)
    print(json.dumps(profile, indent=2, ensure_ascii=False))
    print("Saved to output/profile.json")