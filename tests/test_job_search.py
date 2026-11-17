"""Tests for Agent 2 (job search).

Run from project root:  python -m pytest -v
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "agents"))

from job_search import (  # noqa: E402
    JOB_FIELDS, JobProvider, build_query, experience_level,
    normalize_job, search_jobs,
)
from pipeline import JOB_FIELDS as PIPELINE_JOB_FIELDS  # noqa: E402
from pipeline import run_pipeline, validate_jobs  # noqa: E402

FIXTURES = os.path.join(ROOT, "tests", "fixtures")
RESUME = os.path.join(FIXTURES, "resume_title_case.docx")      # FAKE resume
with open(os.path.join(FIXTURES, "jobs.json"), encoding="utf-8") as f:
    RAW_JOBS = json.load(f)


class FixtureProvider(JobProvider):
    """Fake provider that returns whatever raw jobs it is given."""
    name = "fixture"

    def __init__(self, raw_jobs):
        self.raw_jobs = raw_jobs

    def search(self, query, limit):
        return self.raw_jobs


# ---------- contract and base class ----------

def test_job_fields_match_pipeline_contract():
    # Agent 2 must output exactly what run_pipeline() validates for
    assert set(JOB_FIELDS) == PIPELINE_JOB_FIELDS


def test_base_provider_search_is_not_implemented():
    with pytest.raises(NotImplementedError):
        JobProvider().search({}, 10)


# ---------- normalize_job ----------

def test_ids_are_prefixed_with_source():
    job = normalize_job({"id": "42", "title": "Dev", "description": "x"}, "fixture")
    assert job["id"] == "fixture-42"


def test_missing_id_gets_stable_generated_id():
    raw = {"title": "Dev", "company": "Acme", "description": "x"}
    assert normalize_job(raw, "s")["id"] == normalize_job(raw, "s")["id"]


def test_missing_optional_fields_become_empty_strings():
    job = normalize_job({"title": "Dev", "description": "x"}, "s")
    assert job["company"] == "" and job["location"] == "" and job["url"] == ""


def test_jobs_without_title_or_description_are_dropped():
    raw = [{"title": "", "description": "x"}, {"title": "Dev", "description": ""}]
    assert search_jobs({}, FixtureProvider(raw)) == []


# ---------- search_jobs ----------

def test_output_passes_pipeline_validation():
    jobs = search_jobs({}, FixtureProvider(RAW_JOBS))
    validate_jobs(jobs)  # raises if the shape is wrong
    assert len(jobs) == len(RAW_JOBS)


def test_duplicate_urls_are_removed():
    jobs = search_jobs({}, FixtureProvider(RAW_JOBS + [RAW_JOBS[0]]))
    assert len(jobs) == len(RAW_JOBS)


# A FAKE posting syndicated to two job boards (shaped like the real
# Marsh & McLennan case: same text, different city, url and id)
LISTED_TWICE = [
    {"id": "a1", "title": "React Full Stack - Senior Software Engineer",
     "company": "Fakeco & Partners Companies", "location": "Mandeville, LA",
     "url": "https://board-one.test/job/1", "description": "Build React apps.  Remote role."},
    {"id": "b2", "title": "React Full Stack – Senior Software Engineer",
     "company": "Fakeco & Partners", "location": "Lake Charles, LA",
     "url": "https://board-two.test/job/9", "description": "Build React apps. Remote role."},
]


def test_same_posting_on_two_boards_counts_once():
    jobs = search_jobs({}, FixtureProvider(LISTED_TWICE))
    assert [j["id"] for j in jobs] == ["fixture-a1"]


def test_same_title_company_and_location_counts_once():
    other_text = dict(LISTED_TWICE[1], location="Mandeville, LA", description="Different words.")
    jobs = search_jobs({}, FixtureProvider([LISTED_TWICE[0], other_text]))
    assert len(jobs) == 1


def test_same_title_and_company_elsewhere_with_other_text_is_kept():
    # two real openings: different city AND different description
    other = dict(LISTED_TWICE[1], description="Lead a team in our Lake Charles office.")
    assert len(search_jobs({}, FixtureProvider([LISTED_TWICE[0], other]))) == 2


def test_different_company_or_title_is_kept():
    other_company = dict(LISTED_TWICE[1], company="Othercorp")
    other_title = dict(LISTED_TWICE[1], title="Junior Software Engineer")
    assert len(search_jobs({}, FixtureProvider([LISTED_TWICE[0], other_company]))) == 2
    assert len(search_jobs({}, FixtureProvider([LISTED_TWICE[0], other_title]))) == 2


def test_limit_is_respected():
    assert len(search_jobs({}, FixtureProvider(RAW_JOBS), limit=3)) == 3


# ---------- build_query ----------

def test_experience_level_buckets():
    assert experience_level(None) == "entry"
    assert experience_level(0) == "entry"
    assert experience_level(1.9) == "entry"
    assert experience_level(2) == "mid"
    assert experience_level(4.9) == "mid"
    assert experience_level(5) == "senior"


def test_query_keywords_are_capped_and_deduped():
    profile = {"skills": ["Python", "python", "SQL", "Java", "C++", "Git", "Linux"]}
    query = build_query(profile, max_keywords=4)
    assert query["keywords"] == ["Python", "SQL", "Java", "C++"]


def test_query_from_empty_profile_has_empty_values():
    assert build_query({}) == {
        "keywords": [], "title": "", "location": "", "experience_level": "entry",
    }


def test_query_location_is_passed_through_stripped():
    assert build_query({}, location="  Monroe, LA ")["location"] == "Monroe, LA"


def test_query_title_comes_from_first_experience_entry():
    profile = {"experience": [{"title": "Data Analyst"}, {"title": "Intern"}]}
    assert build_query(profile)["title"] == "Data Analyst"


def test_query_from_real_parsed_profile():
    profile = run_pipeline(RESUME, RAW_JOBS)["profile"]
    query = build_query(profile, location="Monroe, LA")
    assert query["keywords"] and set(query["keywords"]) <= set(profile["skills"])
    assert query["experience_level"] == experience_level(profile["experience_years"])
    assert query["location"] == "Monroe, LA"