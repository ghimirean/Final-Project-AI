"""Tests for Agent 3 (job synthesizer + match scoring) using FAKE jobs.

Run from project root:  python -m pytest -v
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from job_synthesizer import (  # noqa: E402
    load_jobs, extract_requirements, extract_min_years, extract_degree,
    score_match, text_similarity, rank_jobs, make_reason,
)

JOBS = {job["id"]: job for job in load_jobs(os.path.join(ROOT, "tests", "fixtures", "jobs.json"))}

# A FAKE CS student profile in the same shape Agent 1 produces
STUDENT = {
    "name": "Test Student",
    "email": "student@example.com",
    "skills": ["Java", "C++", "Git", "OOP", "Data Structures", "Algorithms"],
    "education": {"degree": "Bachelor of Science in Computer Science",
                  "field": "Computer Science"},
    "experience": [{"company": "Fake College", "title": "Teaching Assistant"}],
    "experience_years": 1.0,
}


# ---------- Step A: input format + requirement extraction ----------

def test_jobs_have_required_fields():
    for job in JOBS.values():
        assert set(job) == {"id", "title", "company", "location", "url", "description"}


def test_java_job_requirements():
    req = extract_requirements(JOBS["job-001"])
    for skill in ["Java", "C++", "Git", "OOP", "Data Structures", "Algorithms"]:
        assert skill in req["skills"]
    assert "JavaScript" not in req["skills"]
    assert req["min_years"] == 0
    assert req["degree"] == "Bachelor"


def test_senior_job_needs_five_years():
    req = extract_requirements(JOBS["job-005"])
    assert req["min_years"] == 5
    assert "Java" in req["skills"] and "AWS" in req["skills"]


def test_ml_job_requirements():
    req = extract_requirements(JOBS["job-004"])
    assert {"Python", "PyTorch", "TensorFlow", "Deep Learning"} <= set(req["skills"])
    assert req["min_years"] == 2
    assert req["degree"] == "Master"


def test_frontend_job_has_no_degree():
    req = extract_requirements(JOBS["job-006"])
    assert req["degree"] is None
    assert req["min_years"] == 1


def test_years_and_degree_helpers():
    assert extract_min_years("3 or more years of work") == 3
    assert extract_min_years("no experience needed") == 0
    assert extract_degree("Bachelor's or Master's degree") == "Bachelor"


# ---------- Step B: match score ----------

def test_score_is_between_0_and_100():
    for job in JOBS.values():
        result = score_match(STUDENT, job)
        assert 0 <= result["score"] <= 100


def test_java_job_beats_ml_job():
    assert score_match(STUDENT, JOBS["job-001"])["score"] > score_match(STUDENT, JOBS["job-004"])["score"]


def test_matched_and_missing_skills():
    b = score_match(STUDENT, JOBS["job-002"])["breakdown"]
    assert set(b["matched_skills"]) == {"Java", "C++", "Git", "Data Structures", "Algorithms"}
    assert set(b["missing_skills"]) == {"GitHub", "Linux"}


def test_experience_penalty_for_senior_role():
    b = score_match(STUDENT, JOBS["job-005"])["breakdown"]
    assert b["required_years"] == 5
    assert b["experience_penalty"] == 30          # 4 years short -> capped at 30
    assert score_match(STUDENT, JOBS["job-001"])["breakdown"]["experience_penalty"] == 0


def test_more_experience_removes_penalty():
    veteran = dict(STUDENT, experience_years=6.0)
    assert score_match(veteran, JOBS["job-005"])["score"] > score_match(STUDENT, JOBS["job-005"])["score"]


def test_text_similarity_range():
    assert text_similarity("java developer", "java developer") > 0.99
    assert text_similarity("java developer", "pastry chef") == 0


# ---------- Step C: ranking + output ----------

def test_rank_jobs_sorted_best_first():
    ranked = rank_jobs(STUDENT, list(JOBS.values()))
    scores = [r["score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
    assert len(ranked) == len(JOBS)


def test_rank_jobs_result_fields():
    for r in rank_jobs(STUDENT, list(JOBS.values())):
        for key in ["id", "title", "company", "score", "matched_skills", "missing_skills", "reason"]:
            assert key in r
        assert isinstance(r["reason"], str) and r["reason"]


def test_java_student_ranking_order():
    ids = [r["id"] for r in rank_jobs(STUDENT, list(JOBS.values()))]
    assert ids[0] == "job-001"                         # Junior Java Developer first
    assert ids.index("job-005") > ids.index("job-001")  # senior role ranks lower


def test_reason_mentions_missing_skills_and_years():
    senior = next(r for r in rank_jobs(STUDENT, list(JOBS.values())) if r["id"] == "job-005")
    assert "needs 5+ yrs" in senior["reason"]
    assert "missing" in senior["reason"]


def test_reason_with_no_skills_in_posting():
    reason = make_reason({"matched_skills": [], "missing_skills": [],
                          "experience_penalty": 0, "required_years": 0,
                          "candidate_years": 1.0})
    assert reason == "No known skills found in the posting."
