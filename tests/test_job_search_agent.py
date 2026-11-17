"""Tests for Agent 2's AI search agent (job_search_agent.py).

No real LLM or job site is ever called: ask_llm is replaced with a scripted
fake and the provider returns canned jobs. All data here is fake.

Run from project root:  python -m pytest -v
"""
import os
import sys

import pytest

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

def test_prompt_asks_for_short_queries_with_common_titles(monkeypatch):
    prompts = script(monkeypatch, FINISH)
    agent.run_job_agent(PROFILE, FakeProvider())
    assert "common job title" in prompts[0]
    assert "1 or 2 skills" in prompts[0]

def test_clean_query_fixes_bad_llm_values():
    q = agent.clean_query({"keywords": ["Python", " python ", "", "A", "B", "C", "D", "E"],
                           "title": " Analyst ", "experience_level": "wizard"},
                          "Remote", "mid")
    assert q["keywords"] == ["Python", "A", "B", "C", "D"]
    assert q["title"] == "Analyst"
    assert q["experience_level"] == "mid"
    assert q["location"] == "Remote"

# ---------- per-search records, errors with status, widening ----------

class RecordingProvider(JobProvider):
    """Like JSearchProvider: sets last_request. `results` maps a location
    to its raw jobs (missing = 0 jobs); `error` makes every search fail."""
    name = "fake"

    def __init__(self, results=None, error=None):
        self.results, self.error, self.locations = results or {}, error, []
        self.last_request = None

    def search(self, query, limit):
        loc = query.get("location", "")
        self.locations.append(loc)
        self.last_request = {"query_text": f"{query['title']} in {loc}", "http_status": 200,
                             "raw_count": None, "from_cache": False}
        if self.error:
            self.last_request["http_status"] = getattr(self.error, "status", None)
            raise self.error
        jobs = self.results.get(loc, [])
        self.last_request["raw_count"] = len(jobs)
        return jobs


def test_each_search_step_records_query_status_and_count(monkeypatch):
    script(monkeypatch, search(["SQL"]), FINISH)
    provider = RecordingProvider({"Remote": [raw_job(1), raw_job(2)]})
    result = agent.run_job_agent(PROFILE, provider, location="Remote")
    record = result["trace"][0]["search"]
    assert record == {"query_text": "Data Analyst in Remote", "location": "Remote",
                      "http_status": 200, "raw_count": 2, "returned": 2, "from_cache": False}


def test_provider_error_is_recorded_with_status_and_logged(monkeypatch, caplog):
    from jsearch_provider import ProviderError
    script(monkeypatch, search(["SQL"]), FINISH)
    error = ProviderError("JSearch rate limit or monthly quota reached (HTTP 429)", 429)
    provider = RecordingProvider(error=error)
    with caplog.at_level("WARNING"):
        result = agent.run_job_agent(PROFILE, provider, location="Monroe, LA")
    step = result["trace"][0]
    assert step["search"]["http_status"] == 429 and "429" in step["search"]["error"]
    assert "429" in step["search_error"]
    assert "Agent 2 search failed" in caplog.text and "429" in caplog.text
    assert provider.locations == ["Monroe, LA"]            # no widening after an error


@pytest.mark.parametrize("location, wider", [
    ("Monroe, LA", ["Louisiana", "Remote"]),
    ("Chicago, IL", ["Illinois", "Remote"]),
    ("Baton Rouge, Louisiana", ["Louisiana", "Remote"]),
    ("Chicago", ["Remote"]),
    ("Toronto, ON", ["Remote"]),
    ("Remote", []),
    ("remote (US)", []),
    ("", []),
])
def test_wider_locations(location, wider):
    assert agent.wider_locations(location) == wider


def test_empty_city_widens_to_state_and_stops_there(monkeypatch):
    script(monkeypatch, search(["Java", "OOP"], title="Software Engineer"), FINISH)
    provider = RecordingProvider({"Louisiana": [raw_job(1)]})
    result = agent.run_job_agent(PROFILE, provider, location="Monroe, LA")
    assert provider.locations == ["Monroe, LA", "Louisiana"]   # Remote not needed
    widen = [s for s in result["trace"] if s["step"] == "widen"]
    assert len(widen) == 1
    assert widen[0]["widened_from"] == "Monroe, LA" and widen[0]["widened_to"] == "Louisiana"
    assert widen[0]["query"]["keywords"] == [] and widen[0]["query"]["title"] == "Software Engineer"
    assert widen[0]["search"]["raw_count"] == 1
    assert [j["id"] for j in result["jobs"]] == ["fake-1"] and result["mode"] == "llm"


def test_empty_state_widens_to_remote(monkeypatch):
    script(monkeypatch, search(["SQL"]), FINISH)
    provider = RecordingProvider({"Remote": [raw_job(1)]})
    result = agent.run_job_agent(PROFILE, provider, location="Monroe, LA")
    assert provider.locations == ["Monroe, LA", "Louisiana", "Remote"]
    widen = [s for s in result["trace"] if s["step"] == "widen"]
    assert [(w["widened_from"], w["widened_to"]) for w in widen] == [
        ("Monroe, LA", "Louisiana"), ("Louisiana", "Remote")]
    assert len(result["jobs"]) == 1


def test_no_widening_when_jobs_found_or_turned_off_or_already_remote(monkeypatch):
    script(monkeypatch, search(["SQL"]), FINISH)
    provider = RecordingProvider({"Monroe, LA": [raw_job(1)]})
    agent.run_job_agent(PROFILE, provider, location="Monroe, LA")
    assert provider.locations == ["Monroe, LA"]

    script(monkeypatch, search(["SQL"]), FINISH)
    provider = RecordingProvider()
    agent.run_job_agent(PROFILE, provider, location="Monroe, LA", widen=False)
    assert provider.locations == ["Monroe, LA"]

    script(monkeypatch, search(["SQL"]), FINISH)
    provider = RecordingProvider()
    result = agent.run_job_agent(PROFILE, provider, location="Remote")
    assert provider.locations == ["Remote"] and result["jobs"] == []


def test_rule_fallback_also_widens(monkeypatch):
    script(monkeypatch, LLMError("groq down"))
    provider = RecordingProvider({"Louisiana": [raw_job(1)]})
    result = agent.run_job_agent(PROFILE, provider, location="Monroe, LA")
    assert result["mode"] == "rules" and provider.locations == ["Monroe, LA", "Louisiana"]
    assert result["trace"][-1]["step"] == "widen" and len(result["jobs"]) == 1


def test_empty_search_tells_the_llm_to_shorten_the_query(monkeypatch):
    prompts = script(monkeypatch, search(["SQL"]), FINISH)
    agent.run_job_agent(PROFILE, RecordingProvider(), location="Remote")
    assert "try the job title alone" in prompts[1]
