"""Compares Agent 1's three parsers: NER (default), Gemini LLM, and rules.

Run from project root:
    python scripts/compare_parsers.py                       # fixtures + data/sample.pdf
    python scripts/compare_parsers.py path/to/resume.pdf    # specific files

Needs the trained NER model (python scripts/train_ner.py) and GEMINI_API_KEY
in .env; one Gemini call per resume. If either is missing, that parser
falls back to rules and the "parsed_by" line says so.

For each resume, prints only the fields where the parsers disagree, then
a summary of how often each pair agrees. Nothing is saved to disk.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from resume_parser import parse_resume, parse_resume_llm, parse_resume_ner  # noqa: E402

SKILLS = os.path.join(ROOT, "data", "skills.txt")
DEFAULT_FILES = [
    "tests/fixtures/resume_title_case.docx",
    "tests/fixtures/resume_numeric_dates.docx",
    "tests/fixtures/resume_comma_skills.docx",
    "data/sample.pdf",
]
PARSERS = [("rules", parse_resume), ("ner", parse_resume_ner), ("llm", parse_resume_llm)]


def fields(profile):
    """The comparable view of a profile. Skills are sorted (order doesn't
    matter), and project summaries are left out (the LLM writes its own)."""
    edu = profile["education"]
    return {
        "name": profile["name"],
        "email": profile["email"],
        "phone": profile["phone"],
        "skills": ", ".join(sorted(profile["skills"])),
        "education": f"{edu['institution']} | {edu['degree']} | {edu['graduation']}",
        "jobs": "; ".join(f"{j['company']} | {j['title']} | {j['months']} mo"
                          for j in profile["experience"]),
        "years": profile["experience_years"],
        "projects": "; ".join(f"{p['name']} | {p['role']} | {p['date']}" for p in profile["projects"]),
    }


def compare(path, agreement):
    results = {name: parse(path, skills_path=SKILLS) for name, parse in PARSERS}
    sources = ", ".join(f"{name} -> parsed_by={r['parsed_by']}" for name, r in results.items())
    print(f"\n=== {os.path.basename(path)}   ({sources})")
    views = {name: fields(r) for name, r in results.items()}
    differences = 0
    for field in views["rules"]:
        values = {name: views[name][field] for name in views}
        for a, b in agreement:
            agreement[(a, b)][0] += values[a] == values[b]
            agreement[(a, b)][1] += 1
        if len(set(map(str, values.values()))) == 1:
            continue
        differences += 1
        print(f"  {field}:")
        for name, value in values.items():
            note = ""
            if field == "skills":
                others = set().union(*(set(views[o]["skills"].split(", ")) for o in views if o != name))
                extra = sorted(set(str(value).split(", ")) - others - {""})
                note = f"   (only {name}: {', '.join(extra)})" if extra else ""
            print(f"    {name:<6} {value}{note}")
    if differences == 0:
        print("  all three parsers agree on every field")


if __name__ == "__main__":
    files = sys.argv[1:] or [f for f in DEFAULT_FILES if os.path.exists(os.path.join(ROOT, f))]
    agreement = {pair: [0, 0] for pair in [("rules", "ner"), ("rules", "llm"), ("ner", "llm")]}
    for f in files:
        compare(os.path.join(ROOT, f) if not os.path.isabs(f) else f, agreement)
    print("\nFields that agree (8 fields per resume):")
    for (a, b), (same, total) in agreement.items():
        print(f"  {a:>5} vs {b:<5} {same}/{total}")
