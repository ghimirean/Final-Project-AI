"""Tests for Agent 1 (resume parser) using the FAKE resumes in tests/fixtures/.

Run from project root:  python -m pytest -v
"""
import os
import sys
from datetime import date

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from resume_parser import parse_resume  # noqa: E402

FIXTURES = os.path.join(ROOT, "tests", "fixtures")
SKILLS = os.path.join(ROOT, "data", "skills.txt")


def parse(filename):
    return parse_resume(os.path.join(FIXTURES, filename), skills_path=SKILLS)


def months_since(year, month):
    today = date.today()
    return today.year * 12 + today.month - (year * 12 + month)


# One row per fake resume: file, name, email, some skills, experience years
CASES = [
    ("resume_title_case.docx", "Priya Nandakumar",
     "priya.nandakumar@example.com", ["Python", "SQL", "Tableau"], 3.2),
    ("resume_numeric_dates.docx", "Marcus Oyelaran",
     "marcus.oyelaran@example.org", ["C++", "Java", "Docker"], 2.8),
    # "2021 - Present" grows over time, so compute it: Jan 2021 -> today,
    # plus 36 months for "2018 - 2021"
    ("resume_comma_skills.docx", "Elena Vasquez-Moreau",
     "elena.vm@example.net", ["JavaScript", "React", "REST API"],
     round((months_since(2021, 1) + 36) / 12, 1)),
]


@pytest.mark.parametrize("filename, name, email, skills, years", CASES)
def test_fake_resume(filename, name, email, skills, years):
    profile = parse(filename)
    assert profile["name"] == name
    assert profile["email"] == email
    for skill in skills:
        assert skill in profile["skills"]
    assert profile["experience_years"] == years


def test_each_resume_finds_two_jobs():
    for filename, *_ in CASES:
        assert len(parse(filename)["experience"]) == 2


def test_unsupported_file_type():
    with pytest.raises(ValueError):
        parse_resume(SKILLS)
