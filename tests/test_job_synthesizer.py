"""Tests for Agent 3 (job synthesizer + match scoring) using FAKE jobs.

Run from project root:  python -m pytest -v
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from job_synthesizer import (  # noqa: E402
    load_jobs, extract_requirements, extract_min_years, extract_degree,
    score_match, job_similarities, rank_jobs, make_reason, resume_text,
    assess_job, rule_based_assessment, candidate_summary, LLM_CONFIG,
)
import llm  # noqa: E402
import pytest  # noqa: E402

JOBS = {job["id"]: job for job in load_jobs(os.path.join(ROOT, "tests", "fixtures", "jobs.json"))}

# A FAKE CS student profile in the same shape Agent 1 produces
STUDENT = {
    "name": "Test Student",
    "email": "student@example.com",
    "skills": ["Java", "C++", "Git", "OOP", "Data Structures", "Algorithms"],
    "education": {"degree": "Bachelor of Science in Computer Science",
                  "field": "Computer Science"},
    "experience": [{"company": "Fake College", "title": "Teaching Assistant"}],
    "experience_years": 1.0,
}


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


# ---------- Step B: match score ----------

def test_score_is_between_0_and_100():
    for job in JOBS.values():
        result = score_match(STUDENT, job)
        assert 0 <= result["score"] <= 100


def test_java_job_beats_ml_job():
    assert score_match(STUDENT, JOBS["job-001"])["score"] > score_match(STUDENT, JOBS["job-004"])["score"]


def test_matched_and_missing_skills():
    b = score_match(STUDENT, JOBS["job-002"])["breakdown"]
    assert set(b["matched_skills"]) == {"Java", "C++", "Git", "Data Structures", "Algorithms"}
    assert set(b["missing_skills"]) == {"GitHub", "Linux"}


def test_experience_penalty_for_senior_role():
    b = score_match(STUDENT, JOBS["job-005"])["breakdown"]
    assert b["required_years"] == 5
    assert b["experience_penalty"] == 30          # 4 years short -> capped at 30
    assert score_match(STUDENT, JOBS["job-001"])["breakdown"]["experience_penalty"] == 0


def test_more_experience_removes_penalty():
    veteran = dict(STUDENT, experience_years=6.0)
    assert score_match(veteran, JOBS["job-005"])["score"] > score_match(STUDENT, JOBS["job-005"])["score"]


def test_job_similarities_rescaled_to_best():
    raw, scaled = job_similarities("java developer",
                                   ["java developer wanted", "pastry chef", "senior java lead"])
    assert max(scaled) == 1.0                      # best job gets the full share
    assert scaled[0] == 1.0 and scaled[1] == 0.0   # chef shares no words
    assert raw[0] > raw[2] > raw[1]


def test_degree_penalty_when_job_needs_higher_degree():
    b = score_match(STUDENT, JOBS["job-004"])["breakdown"]     # asks for a Master's
    assert (b["candidate_degree"], b["required_degree"]) == ("Bachelor", "Master")
    assert b["degree_penalty"] == 5
    assert score_match(STUDENT, JOBS["job-001"])["breakdown"]["degree_penalty"] == 0


def test_no_degree_penalty_when_candidate_has_higher_degree():
    masters = dict(STUDENT, education={"degree": "Master of Science in Computer Science"})
    assert score_match(masters, JOBS["job-004"])["breakdown"]["degree_penalty"] == 0


def test_resume_text_prefers_raw_text():
    assert resume_text(dict(STUDENT, raw_text="full resume text")) == "full resume text"
    assert "Java" in resume_text(STUDENT)          # falls back to the short summary


# ---------- Step C: ranking + output ----------

def test_rank_jobs_sorted_best_first():
    ranked = rank_jobs(STUDENT, list(JOBS.values()))
    scores = [r["score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
    assert len(ranked) == len(JOBS)


def test_rank_jobs_result_fields():
    for r in rank_jobs(STUDENT, list(JOBS.values())):
        for key in ["id", "title", "company", "score", "matched_skills", "missing_skills", "reason"]:
            assert key in r
        assert isinstance(r["reason"], str) and r["reason"]


def test_java_student_ranking_order():
    ids = [r["id"] for r in rank_jobs(STUDENT, list(JOBS.values()))]
    assert ids[0] == "job-001"                         # Junior Java Developer first
    assert ids.index("job-005") > ids.index("job-001")  # senior role ranks lower


def test_reason_mentions_missing_skills_and_years():
    senior = next(r for r in rank_jobs(STUDENT, list(JOBS.values())) if r["id"] == "job-005")
    assert "needs 5+ yrs" in senior["reason"]
    assert "missing" in senior["reason"]


def test_reason_with_no_skills_in_posting():
    reason = make_reason({"matched_skills": [], "missing_skills": [],
                          "experience_penalty": 0, "required_years": 0,
                          "candidate_years": 1.0})
    assert reason == "No known skills found in the posting."


def test_best_text_match_gets_full_text_points():
    ranked = rank_jobs(STUDENT, list(JOBS.values()))
    assert max(r["breakdown"]["text_points"] for r in ranked) == 40.0


def test_ranking_uses_raw_text_when_present():
    # The same student, but the full resume text talks about Python and ML,
    # so the text part should now favor the data science job
    ml_text = dict(STUDENT, raw_text="Python pandas SQL Tableau machine learning "
                                     "scikit-learn data analysis dashboards customer data")
    ranked = {r["id"]: r for r in rank_jobs(ml_text, list(JOBS.values()))}
    assert ranked["job-003"]["breakdown"]["text_points"] == 40.0


# ---------- "MS" is only a Master's degree in a degree context ----------

def test_ms_excel_is_not_a_masters_degree():
    assert extract_degree("Proficient in MS Excel and MS Office.") is None
    assert extract_degree("Bachelor's degree required. Advanced MS Excel skills.") == "Bachelor"


def test_ms_in_degree_context_is_a_masters_degree():
    assert extract_degree("Requires an MS in Computer Science.") == "Master"
    assert extract_degree("M.S. in Data Science preferred.") == "Master"
    assert extract_degree("An M.S. degree is required.") == "Master"
    assert extract_degree("Degree: MS") == "Master"
    assert extract_degree("Master's degree required") == "Master"


def test_bs_ms_pair_counts_lowest_degree():
    # "BS/MS" means a Bachelor's is enough, and the MS is still recognized
    assert extract_degree("BS/MS in Computer Science") == "Bachelor"
    assert extract_degree("B.S. or M.S. in Statistics") == "Bachelor"



# ---------- Step 3: LLM key points (Groq is always faked) ----------

GOOD_ASSESSMENT = {
    "key_points": ["Write internal tools in Java", "Solid OOP and data structures"],
    "required_skills": ["Java", "OOP"],
    "nice_to_have": ["C++"],
    "fit_summary": "Strong fit: the student knows Java, OOP and data structures.",
    "gaps": [],
    "interview_focus": ["OOP design", "Data structures", "A Java project"],
}

PERSON = dict(STUDENT, name="Test Student", email="student@example.com",
              phone="555-000-1111", linkedin="linkedin.com/in/test-student",
              projects=[{"name": "Chess Bot", "role": None, "date": None,
                         "summary": "Wrote a chess engine.", "skills": ["Java"]}])


@pytest.fixture
def fake_groq(monkeypatch):
    """Makes Groq reply with the given dict/text. Returns the list of calls."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    calls = []

    def install(reply):
        text = reply if isinstance(reply, str) else json.dumps(reply)
        def fake_post(url, headers, body):
            calls.append({"url": url, "model": body["model"],
                          "prompt": body["messages"][0]["content"]})
            return {"choices": [{"message": {"content": text}}]}
        monkeypatch.setattr(llm, "_post", fake_post)
        return calls
    return install


def test_llm_key_points(fake_groq):
    fake_groq(GOOD_ASSESSMENT)
    ranked = rank_jobs(STUDENT, [JOBS["job-001"]])
    assert ranked[0]["assessed_by"] == "llm"
    assert ranked[0]["assessment"] == GOOD_ASSESSMENT


def test_llm_does_not_change_the_score(fake_groq):
    # Even if the LLM sends its own "score", it is dropped by the schema
    fake_groq(dict(GOOD_ASSESSMENT, score=5))
    with_llm = rank_jobs(STUDENT, list(JOBS.values()))
    without_llm = rank_jobs(STUDENT, list(JOBS.values()), use_llm=False)
    assert [(r["id"], r["score"]) for r in with_llm] == [(r["id"], r["score"]) for r in without_llm]
    assert "score" not in with_llm[0]["assessment"]


def test_prompt_contents(fake_groq):
    calls = fake_groq(GOOD_ASSESSMENT)
    rank_jobs(PERSON, [JOBS["job-001"]])
    prompt = calls[0]["prompt"]
    assert calls[0]["model"] == LLM_CONFIG["model"]
    assert "api.groq.com" in calls[0]["url"]
    assert "Junior Java Developer" in prompt and "Maplewood" in prompt
    assert "Chess Bot" in prompt                              # projects are included
    assert "score" in prompt and "Do not give your own score" in prompt
    # Contact details are not sent to the LLM
    for private in ["Test Student", "student@example.com", "555-000-1111", "linkedin.com"]:
        assert private not in prompt


def test_candidate_summary_has_no_contact_details():
    summary = candidate_summary(PERSON)
    assert "Java" in summary and "Chess Bot" in summary
    assert "Test Student" not in summary and "student@example.com" not in summary


@pytest.mark.parametrize("bad_reply", [
    "Sure! The candidate looks great.",                                  # not JSON
    {k: v for k, v in GOOD_ASSESSMENT.items() if k != "interview_focus"},  # missing field
    dict(GOOD_ASSESSMENT, key_points=[]),                                # empty key points
    dict(GOOD_ASSESSMENT, fit_summary=""),                               # empty summary
])
def test_bad_llm_reply_falls_back_to_rules(fake_groq, bad_reply):
    fake_groq(bad_reply)
    ranked = rank_jobs(STUDENT, [JOBS["job-001"]])
    assert ranked[0]["assessed_by"] == "rules"
    assert ranked[0]["assessment"]["fit_summary"] == ranked[0]["reason"]


def test_api_down_falls_back_to_rules():
    # conftest.py blocks the network: every job gets rule-based key points
    ranked = rank_jobs(STUDENT, list(JOBS.values()))
    assert {r["assessed_by"] for r in ranked} == {"rules"}


def test_rule_based_assessment_for_senior_role():
    result = score_match(STUDENT, JOBS["job-005"])
    req = extract_requirements(JOBS["job-005"])
    a = rule_based_assessment(JOBS["job-005"], result["breakdown"], req)
    assert set(a) == set(GOOD_ASSESSMENT)                     # same fields as the LLM version
    assert "Experience: 5+ years" in a["key_points"]
    assert any("5+ years" in gap for gap in a["gaps"])
    assert a["interview_focus"][0] == result["breakdown"]["missing_skills"][0]   # gaps first
    assert a["fit_summary"] == make_reason(result["breakdown"])


def test_use_llm_false_makes_no_calls(fake_groq):
    calls = fake_groq(GOOD_ASSESSMENT)
    ranked = rank_jobs(STUDENT, list(JOBS.values()), use_llm=False)
    assert calls == []
    assert {r["assessed_by"] for r in ranked} == {"rules"}


def test_assess_job_returns_source(fake_groq):
    fake_groq(GOOD_ASSESSMENT)
    result = score_match(STUDENT, JOBS["job-001"])
    assessment, source = assess_job(STUDENT, JOBS["job-001"], result)
    assert source == "llm" and assessment["interview_focus"]
