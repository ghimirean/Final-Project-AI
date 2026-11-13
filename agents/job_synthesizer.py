"""Agent 3: job description synthesizer + match scoring.

JOB INPUT FORMAT
----------------
Agent 3 takes a list of job postings. Each posting is a dict:

    {
        "id":          "job-001",                 # unique id (string)
        "title":       "Junior Java Developer",
        "company":     "Maplewood Software",
        "location":    "Monroe, LA",
        "url":         "https://example.com/jobs/job-001",
        "description": "Full job description text ..."
    }

All values are strings. Only "description" is used to find requirements;
the other fields are passed through so they can be shown to the user.
An example list lives in tests/fixtures/jobs.json.
"""
import json
import os
import re

from resume_parser import load_skills, extract_skills

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILLS_PATH = os.path.join(ROOT, "data", "skills.txt")


def load_jobs(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Matches "5+ years", "2 years", "at least 1 year", "3 or more years"
YEARS_PATTERN = r"(\d+)\s*\+?\s*(?:or more\s+)?years?\b"

# Degree levels from lowest to highest, with the words that signal each one
DEGREE_LEVELS = [
    ("Associate", r"\bAssociate'?s?\b"),
    ("Bachelor", r"\bBachelor'?s?\b|\bB\.?S\.?\b|\bB\.?A\.?\b"),
    ("Master", r"\bMaster'?s?\b|\bM\.?S\.?\b|\bMBA\b"),
    ("PhD", r"\bPh\.?D\.?\b|\bDoctorate\b"),
]


def extract_min_years(text):
    # Takes the largest number found, e.g. "5+ years" -> 5. None found -> 0.
    numbers = [int(n) for n in re.findall(YEARS_PATTERN, text, re.IGNORECASE)]
    return max(numbers) if numbers else 0


def extract_degree(text):
    # Returns the LOWEST degree mentioned, since "Bachelor's or Master's"
    # means a Bachelor's is enough. None if no degree is mentioned.
    for level, pattern in DEGREE_LEVELS:
        if re.search(pattern, text):
            return level
    return None


def extract_requirements(job, skill_list=None):
    """Reads a job's description and returns what it asks for:
    {"skills": [...], "min_years": int, "degree": str or None}
    """
    if skill_list is None:
        skill_list = load_skills(SKILLS_PATH)
    text = job["description"]
    return {
        "skills": extract_skills(text, skill_list),
        "min_years": extract_min_years(text),
        "degree": extract_degree(text),
    }
