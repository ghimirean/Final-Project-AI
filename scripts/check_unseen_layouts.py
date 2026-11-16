"""Runs Agent 1 (NER, plus rules for reference) on the 3 fake resumes in
layouts the model never saw (tests/fixtures/unseen/) and lists, field by
field, what it gets right and what breaks.

Run from project root:
    python tests/fixtures/make_unseen_layouts.py     # (re)build the resumes
    python scripts/check_unseen_layouts.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agents"))
sys.path.insert(0, os.path.join(ROOT, "tests", "fixtures"))

from make_unseen_layouts import EXPECTED, OUT  # noqa: E402
from resume_parser import parse_resume, parse_resume_ner  # noqa: E402

SKILLS = os.path.join(ROOT, "data", "skills.txt")


def compare(profile, expected):
    """[(field, ok, got, expected)] for the fields in EXPECTED."""
    rows = []
    for field in ("name", "email"):
        rows.append((field, profile[field] == expected[field], profile[field], expected[field]))
    got = sorted(profile["skills"])
    rows.append(("skills", got == sorted(expected["skills"]), got, sorted(expected["skills"])))
    for key, value in expected["education"].items():
        got = profile["education"].get(key)
        rows.append((f"education.{key}", got == value, got, value))
    got_jobs = [(j["company"], j["title"], j["months"]) for j in profile["experience"]]
    want = expected["experience"]
    same = len(got_jobs) == len(want) and all(
        g[0] == w[0] and g[1] == w[1] and (w[2] is None or g[2] == w[2])
        for g, w in zip(got_jobs, want))
    rows.append(("experience", same, got_jobs, want))
    got = [p["name"] for p in profile["projects"]]
    rows.append(("projects", got == expected["projects"], got, expected["projects"]))
    return rows


if __name__ == "__main__":
    totals = {}
    for filename, expected in EXPECTED.items():
        path = os.path.join(OUT, filename)
        print(f"\n=== {filename}")
        for parser_name, parse in (("ner", parse_resume_ner), ("rules", parse_resume)):
            profile = parse(path, skills_path=SKILLS)
            rows = compare(profile, expected)
            ok = sum(r[1] for r in rows)
            totals[parser_name] = [a + b for a, b in zip(totals.get(parser_name, [0, 0]), [ok, len(rows)])]
            print(f"  [{parser_name}] parsed_by={profile['parsed_by']}: {ok}/{len(rows)} fields right")
            for field, good, got, want in rows:
                if not good:
                    print(f"    BROKEN {field}:\n      got      {got}\n      expected {want}")
    print("\nTOTAL fields right: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in totals.items()))
