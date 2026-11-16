"""Tests for Agent 1 (resume parser) using the FAKE resumes in tests/fixtures/.

Run from project root:  python -m pytest -v
"""
import os
import re
import sys
from datetime import date

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from resume_parser import (  # noqa: E402
    canonical_skill, extract_projects, extract_skills, load_skills, parse_resume, remove_urls,
)

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


def test_profile_keeps_raw_text():
    for filename, name, *_ in CASES:
        profile = parse(filename)
        assert name in profile["raw_text"]
        assert "experience" in profile["raw_text"].lower()


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


# ---------- projects (rule-based) ----------

def test_projects_with_role_and_date():
    projects = parse("resume_title_case.docx")["projects"]
    assert [p["name"] for p in projects] == ["Sales Trend Dashboard", "Library Checkout Analyzer"]
    first = projects[0]
    assert first["role"] == "Team Lead"
    assert first["date"] == "Mar 2022"
    # The first bullet wraps onto a second line; both parts are joined
    assert first["summary"] == ("Built an interactive Tableau dashboard on top of "
                                "SQL queries to show monthly sales trends")
    assert first["skills"] == ["SQL", "Tableau", "pandas"]


def test_project_with_name_only():
    second = parse("resume_title_case.docx")["projects"][1]
    assert (second["role"], second["date"]) == (None, None)
    assert second["skills"] == ["Python"]


def test_project_with_date_range():
    projects = parse("resume_numeric_dates.docx")["projects"]
    assert len(projects) == 1
    assert projects[0]["name"] == "Line-Following Robot"
    assert projects[0]["role"] == "Firmware Developer"
    assert projects[0]["date"] == "09/2023 - 12/2023"
    assert projects[0]["skills"] == ["C++", "Linux"]


def test_no_projects_section_gives_empty_list():
    assert parse("resume_comma_skills.docx")["projects"] == []


def test_project_dates_do_not_count_as_experience():
    # Marcus's project has a date range, but his total stays 3.0 years
    assert parse("resume_numeric_dates.docx")["experience_years"] == 3.0


def test_every_project_has_the_same_fields():
    projects = extract_projects(["Chess Bot | Solo 2023", "- Wrote it in Python"], ["Python"])
    assert projects == [{"name": "Chess Bot", "role": "Solo", "date": "2023",
                         "summary": "Wrote it in Python", "skills": ["Python"]}]


# ---------- skill aliases (data/skill_aliases.txt) ----------

SKILL_LIST = load_skills(os.path.join(ROOT, "data", "skills.txt"))


def test_aliases_give_one_canonical_name():
    text = "Microsoft Excel, MS PowerPoint, Object-Oriented Programming (OOP), RESTful APIs"
    assert extract_skills(text, SKILL_LIST) == ["Excel", "PowerPoint", "OOP", "REST API"]
    assert canonical_skill("object oriented programming") == "OOP"
    assert canonical_skill("Kubernetes") == "Kubernetes"          # unknown names unchanged


def test_short_aliases_need_exact_case():
    assert extract_skills("ML pipelines", SKILL_LIST) == ["Machine Learning"]
    assert extract_skills("html markup", ["Machine Learning"]) == []
    assert extract_skills("I like the ml in html", ["Machine Learning"]) == []


# ---------- planned / future work is not a skill ----------

from resume_parser import future_spans, resume_skills_in_text  # noqa: E402


@pytest.mark.parametrize("text, skills", [
    ("Utilized Git for version control and documented a future roadmap for Docker support.", ["Git"]),
    ("Currently learning Python.", []),
    ("I plan to learn React next year.", []),
    ("Planned: AWS deployment", []),
    ("Built Machine Learning and Deep Learning models in Python.",     # skill names, not plans
     ["Python", "Machine Learning", "Deep Learning"]),
    ("Learning Python. Built a Flask app in Python.", ["Python", "Flask"]),   # also used for real
])
def test_future_work_is_not_a_skill(text, skills):
    assert sorted(resume_skills_in_text(text, SKILL_LIST)) == sorted(skills)


def test_future_span_ends_at_the_sentence():
    text = "Wrote Java tools; future work: Docker. Used SQL daily."
    [(start, end)] = future_spans(text)
    assert text[start:end] == "future work: Docker"


# ---------- academic season dates ----------

from resume_parser import DATE_RANGE, extract_education, extract_experience, to_months  # noqa: E402


@pytest.mark.parametrize("token, month", [
    ("Spring 2024", 1), ("Summer 2024", 6), ("Fall 2024", 9), ("Autumn 2024", 9), ("Winter 2024", 12),
    ("fall 2024", 9),
])
def test_season_is_a_fixed_month(token, month):
    assert to_months(token) == 2024 * 12 + month
    assert to_months(token, is_end=True) == 2024 * 12 + month     # same month as an end date


def test_season_date_ranges():
    jobs, years = extract_experience(["Orchard Valley University | Teaching Assistant",
                                      "Fall 2023 - Spring 2024",
                                      "Robotics Club | Software Lead",
                                      "Summer 2022 – Fall 2023"])
    assert [(j["dates"], j["months"]) for j in jobs] == [("Fall 2023 - Spring 2024", 5),
                                                         ("Summer 2022 – Fall 2023", 16)]
    assert re.search(DATE_RANGE, "Spring 2024 - Present").group(0) == "Spring 2024 - Present"


def test_season_graduation_date():
    edu = extract_education(["Orchard Valley University", "Bachelor of Science in Computer Science",
                             "Expected Spring 2026"])
    assert edu["graduation"] == "Spring 2026"
    assert extract_education(["Some College", "May 2022"])["graduation"] == "May 2022"
