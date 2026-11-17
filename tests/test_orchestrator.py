"""Tests for the INTERVAI orchestrator (agents/orchestrator.py).

Nothing real is called: conftest.py blocks the LLMs and the models, Agent
2's LLM is a scripted fake (as in tests/test_job_search_agent.py) and the
job source is a fake provider. Output goes to a temp folder. All data is fake.
"""
import json
import os

import pytest

import job_search_agent
import orchestrator
from job_search import JobProvider
from orchestrator import run_intervai

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESUME = os.path.join(ROOT, "tests", "fixtures", "resume_title_case.docx")   # FAKE resume
FIXTURE_IDS = {j["id"] for j in json.load(open(os.path.join(ROOT, "tests", "fixtures", "jobs.json")))}


def raw_job(n):
    return {"id": str(n), "title": f"Data Analyst {n}", "company": "Acme",
            "location": "Remote", "url": f"https://x.test/{n}",
            "description": "Analyze data with Python, SQL and Tableau dashboards."}


class FakeProvider(JobProvider):
    name = "fake"

    def __init__(self, jobs=None, error=None, api_key="test-key"):
        self.jobs, self.error, self.api_key, self.calls = jobs or [], error, api_key, 0

    def search(self, query, limit):
        self.calls += 1
        if self.error:
            raise self.error
        return self.jobs


SEARCH = {"thought": "search", "action": "search",
          "query": {"keywords": ["SQL"], "title": "Data Analyst", "location": "", "experience_level": "mid"}}
FINISH = {"thought": "done", "action": "finish", "query": None}


@pytest.fixture
def agent2_llm(monkeypatch):
    """Agent 2's LLM: search once, then finish. Returns the list of prompts."""
    decisions, prompts = [SEARCH, FINISH], []

    def fake(prompt, provider, model, schema=None):
        prompts.append(prompt)
        return decisions.pop(0) if decisions else FINISH
    monkeypatch.setattr(job_search_agent, "ask_llm", fake)
    return prompts


def pick_first(ranked):
    return ranked[0]


def run(tmp_path, **kwargs):
    kwargs.setdefault("choose_job", pick_first)
    return run_intervai(RESUME, "Remote", out_dir=str(tmp_path), **kwargs)


def trace_of(result):
    with open(result["trace_path"], encoding="utf-8") as f:
        return json.load(f)["stages"]


# ---------- the normal flow ----------

def test_agent2_jobs_are_ranked_and_traced(tmp_path, agent2_llm):
    provider = FakeProvider([raw_job(1), raw_job(2)])
    result = run(tmp_path, provider=provider)
    assert result["jobs_source"] == "jsearch"
    assert {r["id"] for r in result["ranked_jobs"]} == {"fake-1", "fake-2"}
    assert result["selected"] == result["ranked_jobs"][0]
    stages = trace_of(result)
    assert [s["agent"].split(":")[0] for s in stages] == [
        "Agent 1", "Agent 2", "Agent 3", "Human in the loop", "Agent 4"]
    agent2 = stages[1]
    assert agent2["fallback_used"] is False and agent2["details"]["jobs_found"] == 2
    assert agent2["kind_of_ai"] == "generative LLM with tool use"
    for s in stages:
        assert {"agent", "kind_of_ai", "model", "seconds", "fallback_used", "fallback"} <= set(s)


def test_stage_fallbacks_are_recorded(tmp_path, agent2_llm):
    # conftest blocks the NER model and the embedding model: both fallbacks show up
    stages = trace_of(run(tmp_path, provider=FakeProvider([raw_job(1)])))
    assert stages[0]["fallback"] == "rule-based parser"
    assert "TF-IDF" in stages[2]["fallback"] and "rule-based key points" in stages[2]["fallback"]


def test_trace_has_no_contact_details(tmp_path, agent2_llm):
    result = run(tmp_path, provider=FakeProvider([raw_job(1)]))
    text = open(result["trace_path"], encoding="utf-8").read()
    for private in ("Priya", "Nandakumar", "priya.nandakumar@example.com", "555", "linkedin"):
        assert private not in text


# ---------- Agent 2 can't search -> fixture jobs, with a warning ----------

@pytest.mark.parametrize("provider, reason", [
    (FakeProvider(api_key=None), "no JSearch key"),
    (FakeProvider(error=RuntimeError("HTTP 429 quota")), "searches failed"),
    (FakeProvider(jobs=[]), "no jobs"),
])
def test_falls_back_to_fixture_jobs(tmp_path, agent2_llm, capsys, provider, reason):
    result = run(tmp_path, provider=provider)
    assert result["jobs_source"] == "fixture"
    assert {r["id"] for r in result["ranked_jobs"]} == FIXTURE_IDS
    assert "WARNING [Orchestrator]" in capsys.readouterr().err
    agent2 = trace_of(result)[1]
    assert agent2["fallback_used"] and reason in agent2["fallback"]


def test_no_key_skips_agent2_entirely(tmp_path, agent2_llm):
    provider = FakeProvider(api_key=None)
    run(tmp_path, provider=provider)
    assert agent2_llm == [] and provider.calls == 0       # no Groq call, no JSearch call


def test_agent2_crash_falls_back(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("JSearch rate limit or monthly quota reached (HTTP 429)")
    monkeypatch.setattr(orchestrator, "run_job_agent", broken)
    result = run(tmp_path, provider=FakeProvider([raw_job(1)]))
    assert result["jobs_source"] == "fixture"
    assert "quota" in trace_of(result)[1]["fallback"]


def test_offline_uses_fixture_jobs_on_purpose(tmp_path, agent2_llm, capsys):
    provider = FakeProvider([raw_job(1)])
    result = run(tmp_path, provider=provider, offline=True)
    assert result["jobs_source"] == "fixture" and provider.calls == 0 and agent2_llm == []
    assert "offline mode" in trace_of(result)[1]["fallback"]
    assert "WARNING [Orchestrator]" not in capsys.readouterr().err   # on purpose: no warning


# ---------- the human choice ----------

def test_choose_job_gets_the_ranking_best_first(tmp_path):
    seen = []
    run(tmp_path, offline=True, choose_job=lambda ranked: seen.append(ranked) or ranked[1])
    scores = [r["score"] for r in seen[0]]
    assert scores == sorted(scores, reverse=True)


def test_quitting_stops_before_agent4(tmp_path):
    result = run(tmp_path, offline=True, choose_job=lambda ranked: None)
    assert result["selected"] is None and result["handoff"] is None
    stages = trace_of(result)
    assert stages[-1]["details"]["quit"] is True
    assert not any(s["agent"].startswith("Agent 4") for s in stages)


def test_choose_job_is_required():
    with pytest.raises(ValueError):
        run_intervai(RESUME, choose_job=None)
