"""Tests for the pipeline entry point and the FastAPI /rank endpoint.

Run from project root:  python -m pytest -v
"""
import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "agents"))

from api import app  # noqa: E402
from pipeline import run_pipeline  # noqa: E402

FIXTURES = os.path.join(ROOT, "tests", "fixtures")
RESUME = os.path.join(FIXTURES, "resume_title_case.docx")      # FAKE resume
with open(os.path.join(FIXTURES, "jobs.json"), encoding="utf-8") as f:
    JOBS = json.load(f)

client = TestClient(app)


# ---------- run_pipeline ----------

def test_pipeline_returns_profile_and_ranked_jobs():
    result = run_pipeline(RESUME, JOBS)
    assert set(result) == {"profile", "ranked_jobs"}
    assert result["profile"]["name"] == "Priya Nandakumar"
    assert "raw_text" not in result["profile"]
    scores = [j["score"] for j in result["ranked_jobs"]]
    assert len(scores) == len(JOBS) and scores == sorted(scores, reverse=True)


def test_pipeline_ranks_data_job_first_for_data_analyst():
    # Priya knows Python, SQL, Tableau, pandas -> Data Science Intern should win
    result = run_pipeline(RESUME, JOBS)
    assert result["ranked_jobs"][0]["id"] == "job-003"


def test_pipeline_rejects_bad_jobs():
    with pytest.raises(ValueError):
        run_pipeline(RESUME, [{"id": "x", "title": "missing description"}])
    with pytest.raises(ValueError):
        run_pipeline(RESUME, [])


# ---------- POST /rank ----------

def post_rank(path, jobs_text, filename=None):
    with open(path, "rb") as f:
        return client.post("/rank",
                           files={"resume": (filename or os.path.basename(path), f)},
                           data={"jobs": jobs_text})


def test_rank_endpoint_ok():
    response = post_rank(RESUME, json.dumps(JOBS))
    assert response.status_code == 200
    body = response.json()
    assert body["profile"]["email"] == "priya.nandakumar@example.com"
    assert "raw_text" not in body["profile"]
    assert len(body["ranked_jobs"]) == len(JOBS)
    assert body["ranked_jobs"][0]["reason"]


def test_rank_endpoint_rejects_wrong_file_type():
    response = post_rank(os.path.join(FIXTURES, "jobs.json"), json.dumps(JOBS))
    assert response.status_code == 400
    assert ".pdf or .docx" in response.json()["detail"]


def test_rank_endpoint_rejects_bad_json():
    response = post_rank(RESUME, "this is not json")
    assert response.status_code == 400


def test_rank_endpoint_rejects_job_missing_fields():
    response = post_rank(RESUME, json.dumps([{"id": "x"}]))
    assert response.status_code == 400
    assert "missing fields" in response.json()["detail"]
