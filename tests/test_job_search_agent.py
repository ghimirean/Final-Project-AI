"""Tests for Agent 2's AI search agent (job_search_agent.py).

No real LLM or job site is ever called: ask_llm is replaced with a scripted
fake and the provider returns canned jobs. All data here is fake.

Run from project root:  python -m pytest -v
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "agents"))

import job_search_agent as agent  # noqa: E402
from job_search import JOB_FIELDS, JobProvider  # noqa: E402
from llm import LLMError  # noqa: E402
from pipeline import validate_jobs  # noqa: E402

PROFILE = {
    "name": "Priya Nandakumar",
    "email": "priya@example.com",
    "phone": "(555) 201-3344",
    "skills": ["Python", "SQL", "Tableau"],
    "experience": [{"company": "Brightline", "title": "Junior Data Analyst",
                    "dates": "Jan 2022 - Dec 2023", "months": 24}],
    "experience_years": 3.3,
    "projects": [{"name": "Sales Dashboard", "skills": ["SQL", "Tableau"]}],
    "raw_text": "SECRET RESUME TEXT",
}


def raw_job(n, title="Data Analyst"):
    return {"id": str(n), "title": f"{title} {n}", "company": "Acme",
            "location": "Remote", "url": f"https://x.test/{n}",
            "description": "Work with data."}


class FakeProvider(JobProvider):
    """Returns one canned batch of raw jobs per search call."""
    name = "fake"

    def __init__(self, *batches):
        self.batches = list(batches)
        self.queries = []

    def search(self, query, limit):
        self.queries.append(query)
        return self.batches.pop(0) if self.batches else []


def search(keywords, title="Data Analyst"):
    return {"thought": "searching", "action": "search",
            "query": {"keywords": keywords, "title": title,
                      "location": "", "experience_level": "mid"}}


FINISH = {"thought": "enough", "action": "finish", "query": None}


def script(monkeypatch, *decisions):
    """Replaces ask_llm with a fake that returns the given decisions in order
    (an Exception in the list is raised). Returns the list of prompts it saw."""
    decisions, prompts = list(decisions), []

    def fake(prompt, provider, model, schema=None):
        prompts.append(prompt)
        d = decisions.pop(0)
        if isinstance(d, Exception):
            raise d
        return d

    monkeypatch.setattr(agent, "ask_llm", fake)
    return prompts


# ---------- the loop ----------

def test_search_then_finish(monkeypatch):
    script(monkeypatch, search(["Python", "SQL"]), FINISH)
    provider = FakeProvider([raw_job(1), raw_job(2)])
    result = agent.run_job_agent(PROFILE, provider, location="Remote")
    assert result["mode"] == "llm"
    assert len(result["jobs"]) == 2
    assert provider.queries[0]["keywords"] == ["Python", "SQL"]
    assert provider.queries[0]["location"] == "Remote"   # caller's location wins


def test_jobs_keep_exact_agent3_format(monkeypatch):
    script(monkeypatch, search(["Python"]), FINISH)
    result = agent.run_job_agent(PROFILE, FakeProvider([raw_job(1)]))
    assert tuple(result["jobs"][0].keys()) == JOB_FIELDS
    assert all(isinstance(v, str) for v in result["jobs"][0].values())


def test_output_passes_pipeline_validation(monkeypatch):
    script(monkeypatch, search(["Python"]), FINISH)
    result = agent.run_job_agent(PROFILE, FakeProvider([raw_job(1), raw_job(2)]))
    validate_jobs(result["jobs"])  # raises if the shape is wrong


def test_weak_results_trigger_second_search_and_merge_without_duplicates(monkeypatch):
    script(monkeypatch, search(["Python", "SQL", "Tableau"]),
           search(["SQL"], title="Analyst"), FINISH)
    provider = FakeProvider([raw_job(1)], [raw_job(1), raw_job(2)])
    result = agent.run_job_agent(PROFILE, provider)
    assert len(provider.queries) == 2
    assert [j["id"] for j in result["jobs"]] == ["fake-1", "fake-2"]


def test_second_prompt_shows_first_search_results(monkeypatch):
    prompts = script(monkeypatch, search(["Python"]), FINISH)
    agent.run_job_agent(PROFILE, FakeProvider([raw_job(1)]))
    assert "Data Analyst 1 - Acme" in prompts[1]


def test_duplicate_query_is_not_rerun(monkeypatch):
    script(monkeypatch, search(["Python", "SQL"]), search(["sql", "PYTHON"]), FINISH)
    provider = FakeProvider([raw_job(1)], [raw_job(2)])
    result = agent.run_job_agent(PROFILE, provider)
    assert len(provider.queries) == 1
    assert len(result["jobs"]) == 1


def test_max_steps_caps_searches(monkeypatch):
    script(monkeypatch, *[search([f"skill{i}"]) for i in range(10)])
    provider = FakeProvider(*[[raw_job(i)] for i in range(10)])
    agent.run_job_agent(PROFILE, provider, max_steps=3)
    assert len(provider.queries) == 3


def test_stops_early_when_limit_reached(monkeypatch):
    prompts = script(monkeypatch, search(["Python"]))
    provider = FakeProvider([raw_job(1), raw_job(2), raw_job(3)])
    result = agent.run_job_agent(PROFILE, provider, limit=2)
    assert len(result["jobs"]) == 2
    assert len(prompts) == 1          # no extra LLM call once the limit is hit


# ---------- fallback and errors ----------

def test_llm_error_on_first_step_falls_back_to_rules(monkeypatch):
    script(monkeypatch, LLMError("GROQ_API_KEY is not set"))
    provider = FakeProvider([raw_job(1)])
    result = agent.run_job_agent(PROFILE, provider, location="Remote")
    assert result["mode"] == "rules"
    assert provider.queries[0]["keywords"] == ["Python", "SQL", "Tableau"]
    assert provider.queries[0]["title"] == "Junior Data Analyst"
    assert len(result["jobs"]) == 1


def test_llm_error_after_results_keeps_them(monkeypatch):
    script(monkeypatch, search(["Python"]), LLMError("busy"))
    result = agent.run_job_agent(PROFILE, FakeProvider([raw_job(1)]))
    assert result["mode"] == "llm"
    assert len(result["jobs"]) == 1


def test_finish_without_searching_falls_back_to_rules(monkeypatch):
    script(monkeypatch, FINISH)
    result = agent.run_job_agent(PROFILE, FakeProvider([raw_job(1)]))
    assert result["mode"] == "rules"


def test_provider_error_does_not_crash(monkeypatch):
    class Broken(JobProvider):
        name = "broken"

        def search(self, query, limit):
            raise RuntimeError("site down")

    script(monkeypatch, search(["Python"]), FINISH)
    result = agent.run_job_agent(PROFILE, Broken())
    assert result["jobs"] == []

     # the failure is recorded in the trace so callers can tell it from "no jobs"
    assert any(step.get("search_error") == "RuntimeError" for step in result["trace"])


# ---------- prompt and query helpers ----------

def test_prompt_has_no_contact_details_or_raw_text(monkeypatch):
    prompts = script(monkeypatch, FINISH)
    agent.run_job_agent(PROFILE, FakeProvider())
    prompt = prompts[0]
    for secret in ("Priya", "priya@example.com", "555", "SECRET RESUME TEXT"):
        assert secret not in prompt
    assert "Python" in prompt and "Sales Dashboard" in prompt


def test_clean_query_fixes_bad_llm_values():
    q = agent.clean_query({"keywords": ["Python", " python ", "", "A", "B", "C", "D", "E"],
                           "title": " Analyst ", "experience_level": "wizard"},
                          "Remote", "mid")
    assert q["keywords"] == ["Python", "A", "B", "C", "D"]
    assert q["title"] == "Analyst"
    assert q["experience_level"] == "mid"
    assert q["location"] == "Remote"