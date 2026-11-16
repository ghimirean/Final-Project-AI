"""Tests for the FastAPI /search endpoint: Agent 1 parses the resume, then
Agent 2 (the LLM search agent) finds jobs.

No real LLM or job site is called: the job provider is a fake and the search
agent's LLM is a scripted fake. Agent 1 uses the same offline fallback as the
/rank tests (the network is blocked in conftest.py), so the resume is parsed
by the rule-based parser. All data here is fake.

Run from project root:  python -m pytest -v
"""
import os
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "agents"))

import api  # noqa: E402
import job_search_agent as agent  # noqa: E402
from job_search import JobProvider  # noqa: E402
from jsearch_provider import ProviderError  # noqa: E402
from llm import LLMError  # noqa: E402
from pipeline import validate_jobs  # noqa: E402

FIXTURES = os.path.join(ROOT, "tests", "fixtures")
RESUME = os.path.join(FIXTURES, "resume_title_case.docx")      # FAKE resume

client = TestClient(api.app)


def raw_job(n):
    return {"id": str(n), "title": f"Data Analyst {n}", "company": "Acme",
            "location": "Remote", "url": f"https://x.test/{n}",
            "description": "Work with data."}


class FakeProvider(JobProvider):
    """Returns one canned batch per search, or raises `error` if given."""
    name = "fake"

    def __init__(self, *batches, error=None):
        self.batches, self.error, self.queries = list(batches), error, []

    def search(self, query, limit):
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.batches.pop(0) if self.batches else []


SEARCH = {"thought": "searching", "action": "search",
          "query": {"keywords": ["Python"], "title": "Data Analyst",
                    "location": "", "experience_level": "mid"}}
FINISH = {"thought": "enough", "action": "finish", "query": None}


def script(monkeypatch, *decisions):
    """Replaces the search agent's ask_llm with scripted decisions
    (an Exception in the list is raised)."""
    decisions = list(decisions)

    def fake(prompt, provider, model, schema=None):
        d = decisions.pop(0)
        if isinstance(d, Exception):
            raise d
        return d

    monkeypatch.setattr(agent, "ask_llm", fake)


def post_search(monkeypatch, provider, path=RESUME, **form):
    monkeypatch.setattr(api, "JSearchProvider", lambda: provider)
    with open(path, "rb") as f:
        return client.post("/search",
                           files={"resume": (os.path.basename(path), f)},
                           data=form)


# ---------- success ----------

def test_search_endpoint_ok(monkeypatch):
    script(monkeypatch, SEARCH, FINISH)
    provider = FakeProvider([raw_job(1), raw_job(2)])
    response = post_search(monkeypatch, provider, location="Louisiana")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"profile", "jobs", "search_mode", "trace"}
    assert body["profile"]["name"] == "Priya Nandakumar"
    assert "raw_text" not in body["profile"]
    assert body["search_mode"] == "llm"
    assert len(body["jobs"]) == 2
    validate_jobs(body["jobs"])                       # ready to send to /rank
    assert provider.queries[0]["location"] == "Louisiana"


def test_search_falls_back_to_rules_when_llm_fails(monkeypatch):
    script(monkeypatch, LLMError("GROQ_API_KEY is not set"))
    response = post_search(monkeypatch, FakeProvider([raw_job(1)]))
    assert response.status_code == 200
    assert response.json()["search_mode"] == "rules"
    assert len(response.json()["jobs"]) == 1


def test_search_with_no_results_returns_empty_list(monkeypatch):
    script(monkeypatch, SEARCH, FINISH)
    response = post_search(monkeypatch, FakeProvider([]))
    assert response.status_code == 200
    assert response.json()["jobs"] == []


# ---------- bad input ----------

def test_search_rejects_wrong_file_type(monkeypatch):
    response = post_search(monkeypatch, FakeProvider(), path=os.path.join(FIXTURES, "jobs.json"))
    assert response.status_code == 400
    assert ".pdf or .docx" in response.json()["detail"]


@pytest.mark.parametrize("limit", ["0", "51"])
def test_search_rejects_bad_limit(monkeypatch, limit):
    response = post_search(monkeypatch, FakeProvider(), limit=limit)
    assert response.status_code == 400
    assert "limit" in response.json()["detail"]


# ---------- job source failures ----------

def test_provider_error_on_rule_fallback_gives_502(monkeypatch):
    script(monkeypatch, LLMError("no key"))
    provider = FakeProvider(error=ProviderError("JSearch rate limit or monthly quota reached (HTTP 429)"))
    response = post_search(monkeypatch, provider)
    assert response.status_code == 502
    assert "quota" in response.json()["detail"]


def test_provider_error_in_llm_path_gives_502(monkeypatch):
    script(monkeypatch, SEARCH, FINISH)
    response = post_search(monkeypatch, FakeProvider(error=RuntimeError("site down")))
    assert response.status_code == 502
    assert "job source" in response.json()["detail"]