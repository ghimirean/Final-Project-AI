"""Compares Agent 1's LLM parser with the rule-based parser.

Run from project root (makes real Gemini calls, one per resume):
    python scripts/compare_parsers.py                       # fixtures + data/sample.pdf
    python scripts/compare_parsers.py path/to/resume.pdf    # specific files

Prints only the fields that differ. Nothing is saved to disk.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from resume_parser import parse_resume, parse_resume_llm  # noqa: E402

SKILLS = os.path.join(ROOT, "data", "skills.txt")
DEFAULT_FILES = [
    "tests/fixtures/resume_title_case.docx",
    "tests/fixtures/resume_numeric_dates.docx",
    "tests/fixtures/resume_comma_skills.docx",
    "data/sample.pdf",
]


def compare(path):
    rules = parse_resume(path, skills_path=SKILLS)
    llm = parse_resume_llm(path, skills_path=SKILLS)
    print(f"\n=== {os.path.basename(path)}  (LLM result parsed_by = {llm['parsed_by']})")
    if llm["parsed_by"] != "llm":
        print("  LLM failed, so both results come from the rules. Nothing to compare.")
        return 0

    differences = 0
    for key in ["name", "email", "phone", "linkedin", "github", "experience_years"]:
        if rules[key] != llm[key]:
            print(f"  {key:<17} rules: {rules[key]!r:<35} llm: {llm[key]!r}")
            differences += 1

    only_llm = [s for s in llm["skills"] if s not in rules["skills"]]
    only_rules = [s for s in rules["skills"] if s not in llm["skills"]]
    if only_llm or only_rules:
        print(f"  skills            only LLM: {only_llm}")
        print(f"                    only rules: {only_rules}")
        differences += 1

    for field in ["institution", "degree", "field", "graduation"]:
        a, b = rules["education"].get(field), llm["education"].get(field)
        if a != b:
            print(f"  education.{field:<7} rules: {a!r:<35} llm: {b!r}")
            differences += 1

    def job_line(j):
        return f"{j['company']} | {j['title']} | {j['dates']} | {j['months']} mo"
    rules_jobs = [job_line(j) for j in rules["experience"]]
    llm_jobs = [job_line(j) for j in llm["experience"]]
    if rules_jobs != llm_jobs:
        print("  experience        rules:")
        for line in rules_jobs:
            print(f"                      {line}")
        print("                    llm:")
        for line in llm_jobs:
            print(f"                      {line}")
        differences += 1

    if differences == 0:
        print("  identical")
    return differences


if __name__ == "__main__":
    files = sys.argv[1:] or [f for f in DEFAULT_FILES if os.path.exists(os.path.join(ROOT, f))]
    for f in files:
        compare(os.path.join(ROOT, f) if not os.path.isabs(f) else f)
