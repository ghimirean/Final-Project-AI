"""Tests for the LLM version of Agent 1. Gemini is always faked: no network."""
import json
import os
from datetime import date

import pytest

import llm
from resume_parser import parse_resume, parse_resume_llm, LLM_CONFIG
from pipeline import run_pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tests", "fixtures")
SKILLS = os.path.join(ROOT, "data", "skills.txt")
RESUME = os.path.join(FIXTURES, "resume_title_case.docx")      # FAKE resume
with open(os.path.join(FIXTURES, "jobs.json"), encoding="utf-8") as f:
    JOBS = json.load(f)

# What a good Gemini answer for the Priya fixture looks like
GOOD_REPLY = {
    "name": "Priya Nandakumar",
    "email": "priya.nandakumar@example.com",
    "phone": "(555) 201-3344",
    "linkedin": "https://www.linkedin.com/in/priya-nandakumar-fake/",
    "github": None,
    "skills": ["python", "SQL", "Tableau", "pandas", "Excel", "Dashboarding"],
    "education": {"institution": "Northfield State University",
                  "degree": "Bachelor of Science in Statistics",
                  "field": "Statistics", "graduation": "May 2022"},
    "experience": [
        {"company": "Brightline Analytics", "title": "Junior Data Analyst",
         "dates": "Jan 2022 - Dec 2023", "start": "2022-01", "end": "2023-12"},
        {"company": "Cedar Grove Library", "title": "Student Assistant",
         "dates": "Sep 2020 - Dec 2021", "start": "2020-09", "end": "2021-12"},
    ],
}


@pytest.fixture
def fake_gemini(monkeypatch):
    """Makes Gemini 'reply' with the given text. Returns the list of calls."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    calls = []

    def install(reply_text):
        def fake_post(url, headers, body):
            calls.append({"url": url, "prompt": body["contents"][0]["parts"][0]["text"]})
            return {"candidates": [{"content": {"parts": [{"text": reply_text}]}}]}
        monkeypatch.setattr(llm, "_post", fake_post)
        return calls
    return install


def parse_llm():
    return parse_resume_llm(RESUME, skills_path=SKILLS)


# ---------- good LLM answer ----------

def test_llm_profile_has_same_format_as_rules(fake_gemini):
    fake_gemini(json.dumps(GOOD_REPLY))
    llm_profile = parse_llm()
    rules_profile = parse_resume(RESUME, skills_path=SKILLS)
    assert list(llm_profile) == list(rules_profile)          # same keys, same order
    assert llm_profile["parsed_by"] == "llm"
    assert rules_profile["parsed_by"] == "rules"


def test_llm_profile_values(fake_gemini):
    fake_gemini(json.dumps(GOOD_REPLY))
    p = parse_llm()
    assert p["name"] == "Priya Nandakumar"
    assert p["linkedin"] == "linkedin.com/in/priya-nandakumar-fake"   # https/www stripped
    assert p["skills"][0] == "Python"            # "python" -> spelling from skills.txt
    assert "Dashboarding" in p["skills"]         # extra skills not in the list are kept
    assert [j["months"] for j in p["experience"]] == [24, 16]   # Python does the math
    assert p["experience_years"] == 3.3
    assert "Brightline Analytics" in p["raw_text"]


def test_prompt_uses_config_model_and_contains_resume(fake_gemini):
    calls = fake_gemini(json.dumps(GOOD_REPLY))
    parse_llm()
    assert f"models/{LLM_CONFIG['model']}:generateContent" in calls[0]["url"]
    assert "Priya Nandakumar" in calls[0]["prompt"]
    assert "Data Structures" in calls[0]["prompt"]           # skill list is included


def test_present_end_date_counts_to_this_month(fake_gemini):
    reply = dict(GOOD_REPLY, experience=[
        {"company": "A", "title": "B", "dates": "2021 - Present", "start": "2021-01", "end": "Present"}])
    fake_gemini(json.dumps(reply))
    today = date.today()
    expected = today.year * 12 + today.month - (2021 * 12 + 1) + 1
    assert parse_llm()["experience"][0]["months"] == expected


# ---------- bad LLM answers -> fall back to rules ----------

def assert_fell_back(profile):
    assert profile["parsed_by"] == "rules"
    assert profile == parse_resume(RESUME, skills_path=SKILLS)


def test_fallback_on_invalid_json(fake_gemini):
    fake_gemini("Sorry, I can't help with that.")
    assert_fell_back(parse_llm())


def test_fallback_on_missing_name(fake_gemini):
    fake_gemini(json.dumps({k: v for k, v in GOOD_REPLY.items() if k != "name"}))
    assert_fell_back(parse_llm())


def test_fallback_on_bad_date_format(fake_gemini):
    reply = dict(GOOD_REPLY, experience=[dict(GOOD_REPLY["experience"][0], start="Jan 2022")])
    fake_gemini(json.dumps(reply))
    assert_fell_back(parse_llm())


def test_fallback_on_end_before_start(fake_gemini):
    reply = dict(GOOD_REPLY, experience=[dict(GOOD_REPLY["experience"][0], start="2024-01", end="2023-01")])
    fake_gemini(json.dumps(reply))
    assert_fell_back(parse_llm())


def test_fallback_when_api_unreachable():
    # conftest.py blocks the network, so this is the "API down" case
    assert_fell_back(parse_llm())


def test_fallback_when_key_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert_fell_back(parse_llm())


# ---------- pipeline uses the LLM by default ----------

def test_pipeline_uses_llm_by_default(fake_gemini):
    fake_gemini(json.dumps(GOOD_REPLY))
    result = run_pipeline(RESUME, JOBS)
    assert result["profile"]["parsed_by"] == "llm"
    assert "raw_text" not in result["profile"]


def test_pipeline_can_skip_llm(fake_gemini):
    calls = fake_gemini(json.dumps(GOOD_REPLY))
    result = run_pipeline(RESUME, JOBS, use_llm=False)
    assert result["profile"]["parsed_by"] == "rules"
    assert calls == []                                        # no API call made
