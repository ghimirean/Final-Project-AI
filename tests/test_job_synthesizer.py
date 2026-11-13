"""Tests for Agent 3 (job synthesizer + match scoring) using FAKE jobs.

Run from project root:  python -m pytest -v
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from job_synthesizer import (  # noqa: E402
    load_jobs, extract_requirements, extract_min_years, extract_degree,
)

JOBS = {job["id"]: job for job in load_jobs(os.path.join(ROOT, "tests", "fixtures", "jobs.json"))}


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
