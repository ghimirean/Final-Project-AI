"""Annotates the 3 hand-written fixture resumes for NER evaluation.

These resumes (tests/fixtures/resume_*.docx) were written before the NER
work and by a different generator, so their layouts are NOT in the
training data. Scoring the model on them shows how well it generalizes
beyond the synthetic training layouts.

The entities are listed by hand below. The script finds every occurrence
of each one in the resume text (whole words only, longest first, no
overlaps) and writes the same format as resumes.jsonl:
    tests/fixtures/ner/handmade_eval.jsonl

Run from project root:
    python tests/fixtures/ner/annotate_handmade.py
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "agents"))
HERE = os.path.dirname(os.path.abspath(__file__))

from resume_parser import read_resume  # noqa: E402

ANNOTATIONS = {
    "resume_title_case.docx": {
        "PERSON": ["Priya Nandakumar"],
        "ORG": ["Brightline Analytics", "Cedar Grove Library", "Northfield State University"],
        "JOB_TITLE": ["Junior Data Analyst", "Student Assistant"],
        "DEGREE": ["Bachelor of Science in Statistics"],
        "DATE": ["Jan 2022", "Dec 2023", "Sep 2020", "Dec 2021", "Mar 2022", "May 2022"],
        "SKILL": ["Tableau", "Excel", "Python", "SQL", "pandas"],
    },
    "resume_numeric_dates.docx": {
        "PERSON": ["Marcus Oyelaran"],
        "ORG": ["Lakeside Robotics", "Harbor Tech Repair", "Westbrook Institute of Technology"],
        "JOB_TITLE": ["Software Engineer Intern", "IT Support Technician"],
        "DEGREE": ["Bachelor of Science in Computer Engineering"],
        "DATE": ["01/2023", "05/2024", "06/2021", "12/2022", "09/2023", "12/2023", "Dec 2024"],
        "SKILL": ["C++", "Java", "Linux", "Docker", "Git"],
    },
    "resume_comma_skills.docx": {
        "PERSON": ["Elena Vasquez-Moreau"],
        "ORG": ["Quillstone Media", "Pinecrest Bakery", "Riverbend Community College"],
        "JOB_TITLE": ["Web Developer", "Shift Lead"],
        "DEGREE": ["Associate of Science in Computer Science"],
        "DATE": ["May 2018", "2021", "2018", "Present"],
        "SKILL": ["JavaScript", "React", "Node.js", "HTML", "CSS", "Git", "REST API"],
    },
}


def find_entities(text, by_label):
    """Every whole-word occurrence of each listed string, longest first, so
    "May 2018" is one DATE and its "2018" isn't labeled a second time."""
    items = sorted(((s, label) for label, strings in by_label.items() for s in strings),
                   key=lambda pair: -len(pair[0]))
    taken, entities = [], []
    for string, label in items:
        pattern = r"(?<![\w+#.])" + re.escape(string) + r"(?![\w+#])"
        for m in re.finditer(pattern, text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue                                  # overlaps a longer entity
            taken.append((m.start(), m.end()))
            entities.append([m.start(), m.end(), label, string])
    return sorted(entities)


if __name__ == "__main__":
    out = os.path.join(HERE, "handmade_eval.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for filename, by_label in ANNOTATIONS.items():
            text = read_resume(os.path.join(ROOT, "tests", "fixtures", filename))
            record = {"id": filename, "text": text, "entities": find_entities(text, by_label)}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{filename}: {len(record['entities'])} entities")
    print("wrote", out)
