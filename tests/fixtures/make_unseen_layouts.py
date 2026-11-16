"""Builds 3 FAKE resumes in layouts Agent 1's NER model never saw in training,
plus the correct answer for each (EXPECTED), to find what breaks.

    tests/fixtures/unseen/two_column.pdf      skills/education left, experience right
    tests/fixtures/unseen/no_headings.docx    no section headings at all
    tests/fixtures/unseen/season_dates.docx   dates like "Fall 2023 - Spring 2024"

All names, emails, phones and companies are invented. Run from project root:
    python tests/fixtures/make_unseen_layouts.py
Check them with: python scripts/check_unseen_layouts.py
No PDF library is needed: the PDF is written by hand (text placed at x/y).
"""
import os

import docx

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "unseen")

# ---------- 1. two columns (PDF) ----------
# Left column (x=50): contact, skills, education. Right column (x=300):
# experience, projects. Lines at the same height sit side by side.
LEFT = [
    ("Jordan Okafor-Bell", 16),
    ("jordan.okafor@example.net", 9),
    ("(555) 404-1212", 9),
    ("", 9),
    ("SKILLS", 11),
    ("Python, SQL, Docker", 9),
    ("Linux, Git", 9),
    ("", 9),
    ("EDUCATION", 11),
    ("Riverbend Technical College", 9),
    ("Associate of Science in", 9),
    ("Information Technology", 9),
    ("May 2023", 9),
]
RIGHT = [
    ("", 16),
    ("", 9),
    ("", 9),
    ("", 9),
    ("EXPERIENCE", 11),
    ("Harborline Logistics | IT Support Specialist", 9),
    ("Jun 2023 - Oct 2025", 9),
    ("- Automated laptop setup with Python scripts", 9),
    ("Copperfield Diner | Line Cook", 9),
    ("Jan 2021 - May 2023", 9),
    ("", 9),
    ("PROJECTS", 11),
    ("Ticket Dashboard | Developer", 9),
    ("- Built a SQL report of open help-desk tickets", 9),
]


def pdf_escape(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_two_column_pdf(path):
    ops, y = [], 760
    for (left, size), (right, _) in zip(LEFT, RIGHT):
        for x, text in ((50, left), (300, right)):
            if text:
                ops.append(f"BT /F1 {size} Tf {x} {y} Td ({pdf_escape(text)}) Tj ET")
        y -= size + 8
    stream = "\n".join(ops).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    data, offsets = b"%PDF-1.4\n", []
    for n, body in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(data)
    data += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    data += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    data += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    with open(path, "wb") as f:
        f.write(data)


# ---------- 2. no section headings (docx) ----------
NO_HEADINGS = [
    "Mei-Ling Castellanos",
    "meiling.castellanos@example.org | (555) 733-0198",
    "Front-end developer who builds accessible websites for nonprofits.",
    "Lumen Web Studio | Front-End Developer",
    "Mar 2022 - Present",
    "- Built React and TypeScript sites for 12 nonprofit clients",
    "Brightwater Bookshop | Sales Associate",
    "Jun 2019 - Feb 2022",
    "Comfortable with JavaScript, HTML, CSS and Git.",
    "Bachelor of Arts in Graphic Design, Eastvale University, May 2021",
]

# ---------- 3. season dates (docx) ----------
SEASON_DATES = [
    "Tobias Wren",
    "tobias.wren@example.com | (555) 860-4471",
    "Experience",
    "Orchard Valley University | Teaching Assistant",
    "Fall 2023 - Spring 2024",
    "- Graded Java assignments and held weekly lab hours",
    "Northgate Robotics Club | Software Lead",
    "Summer 2022 - Fall 2023",
    "- Wrote C++ motor control code",
    "Skills",
    "Java, C++, Python, Git",
    "Education",
    "Orchard Valley University",
    "Bachelor of Science in Computer Science",
    "Expected Spring 2026",
]

# The correct profile for each resume (the fields the parsers produce)
EXPECTED = {
    "two_column.pdf": {
        "name": "Jordan Okafor-Bell", "email": "jordan.okafor@example.net",
        "skills": ["Python", "SQL", "Docker", "Linux", "Git"],
        "education": {"institution": "Riverbend Technical College",
                      "degree": "Associate of Science in Information Technology",
                      "graduation": "May 2023"},
        "experience": [("Harborline Logistics", "IT Support Specialist", 29),
                       ("Copperfield Diner", "Line Cook", 29)],
        "projects": ["Ticket Dashboard"],
    },
    "no_headings.docx": {
        "name": "Mei-Ling Castellanos", "email": "meiling.castellanos@example.org",
        "skills": ["JavaScript", "TypeScript", "HTML", "CSS", "React", "Git"],
        "education": {"institution": "Eastvale University",
                      "degree": "Bachelor of Arts in Graphic Design", "graduation": "May 2021"},
        "experience": [("Lumen Web Studio", "Front-End Developer", None),   # "Present": depends on today
                       ("Brightwater Bookshop", "Sales Associate", 33)],
        "projects": [],
    },
    "season_dates.docx": {
        "name": "Tobias Wren", "email": "tobias.wren@example.com",
        "skills": ["Java", "C++", "Python", "Git"],
        "education": {"institution": "Orchard Valley University",
                      "degree": "Bachelor of Science in Computer Science",
                      "graduation": "Spring 2026"},     # "Expected" isn't part of the date
        # Seasons are fixed months: Spring = Jan, Summer = Jun, Fall = Sep, Winter = Dec,
        # so Sep 2023 - Jan 2024 = 5 months and Jun 2022 - Sep 2023 = 16 months
        "experience": [("Orchard Valley University", "Teaching Assistant", 5),
                       ("Northgate Robotics Club", "Software Lead", 16)],
        "projects": [],
    },
}


def write_docx(path, lines):
    document = docx.Document()
    for line in lines:
        document.add_paragraph(line)
    document.save(path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    write_two_column_pdf(os.path.join(OUT, "two_column.pdf"))
    write_docx(os.path.join(OUT, "no_headings.docx"), NO_HEADINGS)
    write_docx(os.path.join(OUT, "season_dates.docx"), SEASON_DATES)
    print(f"wrote 3 fake resumes to {os.path.relpath(OUT)}/")
