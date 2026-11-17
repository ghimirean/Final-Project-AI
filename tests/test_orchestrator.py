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


# ---------- HITL: the terminal chooser ----------

from orchestrator import format_choices, make_terminal_chooser  # noqa: E402


def chooser(*answers):
    """A terminal chooser that 'types' the given answers. Returns (choose, shown)."""
    answers, shown = list(answers), []

    def ask(prompt):
        if not answers:
            raise EOFError
        return answers.pop(0)
    return make_terminal_chooser(ask=ask, show=shown.append), shown


def ranked_jobs(tmp_path):
    return run(tmp_path, offline=True)["ranked_jobs"]


def test_menu_shows_top_5_with_score_company_and_gaps(tmp_path):
    ranked = ranked_jobs(tmp_path)
    menu = format_choices(ranked)
    assert menu.count("gaps:") == 5                          # 6 jobs, only the top 5 shown
    first = ranked[0]
    assert f"1. [{first['score']:5.1f}] {first['title']} - {first['company']}" in menu
    assert "'c' for a custom role" in menu


def test_pick_a_number(tmp_path):
    ranked = ranked_jobs(tmp_path)
    choose, _ = chooser("2")
    assert choose(ranked) is ranked[1]


@pytest.mark.parametrize("bad", ["0", "6", "abc", "", "-1", "2.5"])
def test_bad_input_asks_again(tmp_path, bad):
    ranked = ranked_jobs(tmp_path)
    choose, shown = chooser(bad, "1")
    assert choose(ranked) is ranked[0]
    assert any("Please type a number from 1 to 5" in s for s in shown)


def test_custom_role_is_validated(tmp_path):
    ranked = ranked_jobs(tmp_path)
    choose, shown = chooser("c", "", "c", "12345", "c", "Data Engineer")
    assert choose(ranked) == {"custom_role": "Data Engineer"}
    assert any("can't be empty" in s for s in shown)
    assert any("Please type a job title" in s for s in shown)


@pytest.mark.parametrize("answers", [("q",), ("QUIT",), ()])        # () = Ctrl-D
def test_quit(tmp_path, answers):
    choose, _ = chooser(*answers)
    assert choose(ranked_jobs(tmp_path)) is None


def test_no_jobs_to_choose():
    choose, shown = chooser("1")
    assert choose([]) is None and "No jobs" in shown[0]


def test_custom_role_goes_through_agent3(tmp_path):
    result = run(tmp_path, offline=True, choose_job=lambda ranked: {"custom_role": "Data Engineer"})
    selected = result["selected"]
    assert selected["id"] == "custom-role" and selected["title"] == "Data Engineer"
    assert {"score", "assessment", "assessed_by"} <= set(selected)    # same fields as a ranked job
    hitl = [s for s in trace_of(result) if s["agent"].startswith("Human")][0]
    assert hitl["details"]["custom_role"] is True


# ---------- Agent 4 handoff: the interview brief ----------

from orchestrator import build_brief, scrub  # noqa: E402

PRIVATE = ("Priya", "Nandakumar", "priya.nandakumar@example.com", "(555) 201-3344",
           "201-3344", "linkedin", "http")


def brief_of(result):
    with open(result["handoff"]["brief"], encoding="utf-8") as f:
        return json.load(f)


def test_brief_has_the_job_and_the_candidate(tmp_path):
    result = run(tmp_path, offline=True)
    brief = brief_of(result)
    selected = result["selected"]
    job = brief["job"]
    assert job["title"] == selected["title"] and job["company"] == selected["company"]
    jobs = {j["id"]: j for j in json.load(open(os.path.join(ROOT, "tests", "fixtures", "jobs.json")))}
    assert job["description"] == jobs[selected["id"]]["description"]      # the original posting
    for field in ("key_points", "required_skills", "gaps", "interview_focus"):
        assert job[field] == selected["assessment"][field]
    candidate = brief["candidate"]
    assert set(candidate) == {"skills", "projects", "experience_years", "education"}
    assert candidate["skills"] == result["profile"]["skills"]
    assert [p["name"] for p in candidate["projects"]] == [p["name"] for p in result["profile"]["projects"]]
    assert brief["jobs_source"] == "fixture"
    agent4 = trace_of(result)[-1]
    assert agent4["details"]["brief"].endswith("interview_brief.json")


def test_brief_has_no_contact_details(tmp_path):
    result = run(tmp_path, offline=True)
    assert result["profile"]["name"] == "Priya Nandakumar" and result["profile"]["linkedin"]
    text = open(result["handoff"]["brief"], encoding="utf-8").read()
    for private in PRIVATE:
        assert private not in text


def test_free_text_is_scrubbed():
    profile = {"name": "Ann Lee", "skills": ["Python"], "experience_years": 1.0,
               "education": {"degree": "BS", "email": "x"},
               "projects": [{"name": "Ann Lee's Bot", "role": None, "date": None, "skills": [],
                             "summary": "Code at github.com/annlee/bot, ask ann@example.com or 555-123-4567"}]}
    selected = {"id": "j", "title": "Dev", "company": "Acme", "score": 50.0,
                "assessment": {"key_points": [], "required_skills": [], "nice_to_have": [],
                               "gaps": [], "interview_focus": [
                                   "Git (ask for an example; resume: Portfolio: github.com/annlee)"]}}
    brief = build_brief(selected, "Email jobs@acme.example.com to apply.", profile, "jsearch")
    text = json.dumps(brief)
    for private in ("Ann Lee", "github.com", "ann@example.com", "555-123-4567", "jobs@acme"):
        assert private not in text
    assert brief["candidate"]["projects"][0]["name"] == "the candidate's Bot"
    assert brief["candidate"]["education"] == {"institution": None, "degree": "BS",
                                               "field": None, "graduation": None}


def test_scrub_keeps_normal_text():
    assert scrub("Built a Java app used by 500 students in 2024.") == \
        "Built a Java app used by 500 students in 2024."
    assert scrub(None) is None


def test_no_brief_when_the_user_quits(tmp_path):
    run(tmp_path, offline=True, choose_job=lambda ranked: None)
    assert not os.path.exists(os.path.join(tmp_path, "interview_brief.json"))


def test_custom_role_brief(tmp_path):
    result = run(tmp_path, offline=True, choose_job=lambda ranked: {"custom_role": "Data Engineer"})
    job = brief_of(result)["job"]
    assert job["title"] == "Data Engineer" and job["description"] == "Data Engineer."


def test_agent2_llm_failure_is_a_recorded_fallback(tmp_path, monkeypatch, capsys):
    # Groq rejects the model: Agent 2 still searches with its rule-based query.
    # The jobs are real (not the fixture), but the trace must say the LLM didn't run.
    from llm import LLMError

    def model_gone(prompt, provider, model, schema=None):
        raise LLMError("API returned HTTP 404: model_not_found")
    monkeypatch.setattr(job_search_agent, "ask_llm", model_gone)
    result = run(tmp_path, provider=FakeProvider([raw_job(1)]))
    assert result["jobs_source"] == "jsearch"
    agent2 = trace_of(result)[1]
    assert agent2["fallback_used"] and "rule-based search query" in agent2["fallback"]
    assert agent2["details"]["search_mode"] == "rules"
    assert "model_not_found" in agent2["details"]["llm_error"]
    assert "WARNING [Orchestrator]: Agent 2's LLM failed" in capsys.readouterr().err


# ---------- Agent 2 searches in the trace; real vs fixture jobs ----------

class StatusProvider(FakeProvider):
    """FakeProvider that also sets last_request like JSearchProvider."""

    def __init__(self, by_location=None, error=None):
        super().__init__(error=error)
        self.by_location, self.locations = by_location or {}, []

    def search(self, query, limit):
        self.calls += 1
        loc = query.get("location", "")
        self.locations.append(loc)
        self.last_request = {"query_text": f"{query['title']} in {loc}",
                             "http_status": getattr(self.error, "status", 200),
                             "raw_count": None, "from_cache": False}
        if self.error:
            raise self.error
        jobs = self.by_location.get(loc, [])
        self.last_request["raw_count"] = len(jobs)
        return jobs


def run_at(tmp_path, location, **kwargs):
    return run_intervai(RESUME, location, choose_job=pick_first, out_dir=str(tmp_path), **kwargs)


def test_trace_lists_each_search_and_real_job_count(tmp_path, agent2_llm):
    result = run(tmp_path, provider=StatusProvider({"Remote": [raw_job(1), raw_job(2)]}))
    details = trace_of(result)[1]["details"]
    assert details["real_jobs"] == 2 and details["fixture_jobs"] == 0
    assert details["searches"] == [{"step": 1, "query_text": "Data Analyst in Remote",
                                    "location": "Remote", "http_status": 200, "raw_count": 2,
                                    "returned": 2, "from_cache": False}]


def test_fixture_jobs_are_counted_apart_from_real_jobs(tmp_path, agent2_llm):
    result = run(tmp_path, provider=StatusProvider())          # 0 jobs everywhere
    details = trace_of(result)[1]["details"]
    assert details["real_jobs"] == 0 and details["fixture_jobs"] == len(FIXTURE_IDS)
    assert details["searches"][0]["http_status"] == 200 and details["searches"][0]["raw_count"] == 0


def test_widening_is_traced_and_warned(tmp_path, agent2_llm, capsys):
    provider = StatusProvider({"Louisiana": [raw_job(1)]})
    result = run_at(tmp_path, "Monroe, LA", provider=provider)
    assert result["jobs_source"] == "jsearch" and provider.locations == ["Monroe, LA", "Louisiana"]
    details = trace_of(result)[1]["details"]
    assert details["widened_to"] == ["Louisiana"] and details["real_jobs"] == 1
    widen = details["searches"][-1]
    assert widen["step"] == "widen" and widen["widened_from"] == "Monroe, LA"
    assert widen["location"] == "Louisiana" and widen["raw_count"] == 1
    assert "widened to: Louisiana" in capsys.readouterr().err


def test_search_error_warning_shows_the_status_code(tmp_path, agent2_llm, capsys):
    from jsearch_provider import ProviderError
    error = ProviderError("JSearch rate limit or monthly quota reached (HTTP 429)", 429)
    result = run_at(tmp_path, "Monroe, LA", provider=StatusProvider(error=error))
    err = capsys.readouterr().err
    assert "WARNING [Orchestrator]: Agent 2 search failed (HTTP 429)" in err
    agent2 = trace_of(result)[1]
    assert "HTTP 429" in agent2["fallback"] and "found no jobs" not in agent2["fallback"]
    assert agent2["details"]["searches"][0]["http_status"] == 429
