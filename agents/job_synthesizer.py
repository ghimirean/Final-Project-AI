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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


# ---------- Match scoring ----------

# WEIGHTS (total = 100 points)
# - Skills (60 pts): the most direct signal. Recruiters screen by "do you
#   have the listed skills", so the share of required skills you have
#   counts the most.
# - Text similarity (40 pts): catches overlap the skill list misses
#   (job titles, field of study, words like "backend" or "data"). It is
#   noisier than an exact skill match, so it gets less weight.
# - Experience penalty: minus 10 pts per year you are short of the job's
#   minimum, capped at 30, so a senior role drops down the list without
#   being thrown out completely.
SKILL_WEIGHT = 60
TEXT_WEIGHT = 40
PENALTY_PER_YEAR = 10
MAX_PENALTY = 30


def profile_to_text(profile):
    # Agent 1's profile has no raw text, so build a short summary from it
    parts = list(profile.get("skills", []))
    for job in profile.get("experience", []):
        parts += [job.get("title") or "", job.get("company") or ""]
    education = profile.get("education") or {}
    parts += [education.get("degree") or "", education.get("field") or ""]
    return " ".join(p for p in parts if p)


def text_similarity(text_a, text_b):
    # TF-IDF turns each text into a vector of word weights (rare, specific
    # words count more than common ones). Cosine similarity then measures
    # how much the two vectors point the same way: 0 = nothing shared, 1 = same.
    vectors = TfidfVectorizer(stop_words="english").fit_transform([text_a, text_b])
    return float(cosine_similarity(vectors[0], vectors[1])[0][0])


def score_match(profile, job, skill_list=None):
    """Scores how well a resume profile fits one job, from 0 to 100.
    Returns {"score": float, "breakdown": {...}} so the score can be explained.
    """
    req = extract_requirements(job, skill_list)
    have = set(profile.get("skills", []))
    matched = [s for s in req["skills"] if s in have]
    missing = [s for s in req["skills"] if s not in have]

    skill_ratio = len(matched) / len(req["skills"]) if req["skills"] else 0.0
    similarity = text_similarity(profile_to_text(profile), job["description"])

    years = profile.get("experience_years", 0)
    years_short = max(0, req["min_years"] - years)
    penalty = min(MAX_PENALTY, PENALTY_PER_YEAR * years_short)

    skill_points = SKILL_WEIGHT * skill_ratio
    text_points = TEXT_WEIGHT * similarity
    score = max(0.0, min(100.0, skill_points + text_points - penalty))

    return {
        "score": round(score, 1),
        "breakdown": {
            "matched_skills": matched,
            "missing_skills": missing,
            "skill_points": round(skill_points, 1),
            "text_similarity": round(similarity, 3),
            "text_points": round(text_points, 1),
            "required_years": req["min_years"],
            "candidate_years": years,
            "experience_penalty": round(penalty, 1),
        },
    }
