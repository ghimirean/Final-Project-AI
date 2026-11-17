"""Tests for the two halves of the orchestrator run, prepare_run() and
finish_run(), which a web dashboard uses to pause for the job choice.

Offline mode only (the fake fixture jobs), so no job source or LLM is
called; conftest.py blocks the models. Output goes to a temp folder. All
data is fake.
"""
import json
import os

import pytest

from orchestrator import finish_run, prepare_run, run_intervai

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESUME = os.path.join(ROOT, "tests", "fixtures", "resume_title_case.docx")   # FAKE resume


def prepare(tmp_path, **kwargs):
    return prepare_run(RESUME, "Remote", offline=True, out_dir=str(tmp_path), **kwargs)


def stages_of(result):
    with open(result["trace_path"], encoding="utf-8") as f:
        return json.load(f)["stages"]


def brief_path(tmp_path):
    return os.path.join(str(tmp_path), "interview_brief.json")


# ---------- prepare_run: Agents 1-3, stops before the choice ----------

def test_prepare_run_ranks_jobs_and_stops_before_the_choice(tmp_path):
    state = prepare(tmp_path)
    assert state["jobs_source"] == "fixture"
    assert state["ranked_jobs"]
    scores = [r["score"] for r in state["ranked_jobs"]]
    assert scores == sorted(scores, reverse=True)
    assert os.path.exists(os.path.join(str(tmp_path), "ranked_jobs.json"))
    # nothing about the choice is written yet
    assert not os.path.exists(brief_path(tmp_path))
    assert not os.path.exists(os.path.join(str(tmp_path), "run_trace.json"))
    assert [s["agent"] for s in state["trace"].stages] == [
        "Agent 1: resume parser", "Agent 2: job search", "Agent 3: ranking and key points"]


def test_prepare_run_rejects_offline_and_demo_together(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        prepare(tmp_path, demo=True)


# ---------- finish_run: the choice, the brief and the trace ----------

def test_chosen_job_writes_the_brief_and_full_trace(tmp_path):
    state = prepare(tmp_path)
    chosen = state["ranked_jobs"][1]
    result = finish_run(state, chosen)
    assert result["selected"]["id"] == chosen["id"]
    assert result["handoff"]["brief"] == brief_path(tmp_path)
    with open(brief_path(tmp_path), encoding="utf-8") as f:
        brief = json.load(f)
    assert brief["job"]["id"] == chosen["id"]
    assert brief["jobs_source"] == "fixture"
    stages = stages_of(result)
    assert [s["agent"] for s in stages] == [
        "Agent 1: resume parser", "Agent 2: job search", "Agent 3: ranking and key points",
        "Human in the loop: job choice", "Agent 4: interview (handoff)"]
    hitl = stages[3]["details"]
    assert hitl == {"choice": chosen["id"], "title": chosen["title"],
                    "custom_role": False, "quit": False}
    assert "raw_text" not in result["profile"]


def test_custom_role_is_scored_and_handed_off(tmp_path):
    state = prepare(tmp_path)
    result = finish_run(state, {"custom_role": "Data Engineer"})
    assert result["selected"]["id"] == "custom-role"
    assert result["selected"]["title"] == "Data Engineer"
    assert "score" in result["selected"]
    with open(brief_path(tmp_path), encoding="utf-8") as f:
        brief = json.load(f)
    assert brief["job"]["title"] == "Data Engineer"
    assert brief["job"]["description"] == "Data Engineer."
    hitl = stages_of(result)[3]
    assert hitl["agent"] == "Human in the loop: job choice"
    assert hitl["details"]["custom_role"] is True


def test_quit_writes_no_brief(tmp_path):
    state = prepare(tmp_path)
    result = finish_run(state, None)
    assert result["selected"] is None and result["handoff"] is None
    assert not os.path.exists(brief_path(tmp_path))
    stages = stages_of(result)
    assert stages[-1]["agent"] == "Human in the loop: job choice"
    assert stages[-1]["details"]["quit"] is True
    assert all(s["agent"] != "Agent 4: interview (handoff)" for s in stages)


def test_a_run_can_only_be_finished_once(tmp_path):
    state = prepare(tmp_path)
    finish_run(state, None)
    with pytest.raises(ValueError, match="already finished"):
        finish_run(state, state["ranked_jobs"][0])


# ---------- run_intervai is prepare_run + finish_run ----------

def test_run_intervai_matches_the_two_halves(tmp_path):
    seen = []

    def choose(ranked):
        seen.append([r["id"] for r in ranked])
        return ranked[0]

    whole = run_intervai(RESUME, "Remote", choose_job=choose, offline=True,
                         out_dir=str(tmp_path / "whole"))
    state = prepare(tmp_path / "halves")
    halves = finish_run(state, state["ranked_jobs"][0])
    assert seen == [[r["id"] for r in state["ranked_jobs"]]]
    assert whole["selected"]["id"] == halves["selected"]["id"]
    assert [s["agent"] for s in stages_of(whole)] == [s["agent"] for s in stages_of(halves)]
    assert set(whole) == set(halves)
