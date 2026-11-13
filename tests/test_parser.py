"""Tests for Agent 1 (resume parser) using the FAKE resumes in tests/fixtures/.

Run from project root:  python -m pytest -v
"""
import os
import sys
from datetime import date

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from resume_parser import parse_resume, remove_urls  # noqa: E402

FIXTURES = os.path.join(ROOT, "tests", "fixtures")
SKILLS = os.path.join(ROOT, "data", "skills.txt")
SAMPLE_PDF = os.path.join(ROOT, "data", "sample.pdf")


def parse(filename):
    return parse_resume(os.path.join(FIXTURES, filename), skills_path=SKILLS)


def months_since(year, month):
    today = date.today()
    return today.year * 12 + today.month - (year * 12 + month)


# One row per fake resume: file, name, email, some skills, experience years.
# Months are counted inclusively (Jan-Dec 2023 = 12).
CASES = [
    # Jan 2022-Dec 2023 = 24, Sep 2020-Dec 2021 = 16 -> 40 months
    ("resume_title_case.docx", "Priya Nandakumar",
     "priya.nandakumar@example.com", ["Python", "SQL", "Tableau"], 3.3),
    # 01/2023-05/2024 = 17, 06/2021-12/2022 = 19 -> 36 months
    ("resume_numeric_dates.docx", "Marcus Oyelaran",
     "marcus.oyelaran@example.org", ["C++", "Java", "Docker"], 3.0),
    # "2021 - Present" grows over time, so compute it: Jan 2021 -> this month,
    # plus 48 months for "2018 - 2021" (Jan 2018-Dec 2021)
    ("resume_comma_skills.docx", "Elena Vasquez-Moreau",
     "elena.vm@example.net", ["JavaScript", "React", "REST API"],
     round((months_since(2021, 1) + 1 + 48) / 12, 1)),
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


def test_job_months_are_inclusive():
    jobs = parse("resume_title_case.docx")["experience"]
    assert [j["months"] for j in jobs] == [24, 16]


def test_profile_links_are_not_skills():
    # Marcus has "GitHub: github.com/..." but doesn't list GitHub as a skill
    assert "GitHub" not in parse("resume_numeric_dates.docx")["skills"]
    assert "GitHub" not in parse("resume_title_case.docx")["skills"]


def test_remove_urls_keeps_skill_names():
    text = "Node.js, React | GitHub: github.com/me https://site.io/x www.a.com"
    cleaned = remove_urls(text)
    assert "Node.js" in cleaned and "React" in cleaned
    assert "github" not in cleaned.lower() and "http" not in cleaned


# sample.pdf is a real resume kept out of git, so skip this test if it's missing
@pytest.mark.skipif(not os.path.exists(SAMPLE_PDF), reason="data/sample.pdf not present")
def test_sample_pdf_experience_years():
    profile = parse_resume(SAMPLE_PDF, skills_path=SKILLS)
    # Aug 2024 - Present, plus Nov 2021 - Jan 2022 (3 months)
    expected = round((months_since(2024, 8) + 1 + 3) / 12, 1)
    assert profile["experience_years"] == expected
    # GitHub IS listed under "Tools & Software", so it should still be found
    assert "GitHub" in profile["skills"]


def test_unsupported_file_type():
    with pytest.raises(ValueError):
        parse_resume(SKILLS)
