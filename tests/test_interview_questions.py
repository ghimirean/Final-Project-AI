"""Tests for Agent 4, step 1: reading the interview brief and building the
questions (agents/interview_questions.py, run_interview in agents/interview_agent.py).
All data is fake. Nothing is recorded and no model is loaded."""
import json

import pytest

import interview_agent
from interview_questions import BriefError, build_questions, load_brief


def make_brief(**job_overrides):
    job = {"id": "job-1", "title": "Junior Java Developer", "company": "Maplewood Software",
           "location": "Monroe, LA", "score": 90.0, "description": "Build tools in Java.",
           "key_points": ["Java"], "required_skills": ["Java", "Git"], "nice_to_have": [],
           "gaps": ["Linux"],
           "interview_focus": ["Linux (asked for, not on the resume)",
                               "Java (ask for an example; resume: skills: Java)"]}
    job.update(job_overrides)
    return {"brief_version": 1, "created": "2026-11-18T10:00:00", "jobs_source": "offline", "job": job,
            "candidate": {"skills": ["Java"], "projects": [{"name": "Car Rental System", "role": "Lead Developer",
                                                              "date": "Jun 2024", "summary": "x", "skills": ["Java"]}],
                          "experience_years": 1.0, "education": {}}}


def write(tmp_path, brief, name="interview_brief.json"):
    path = tmp_path / name
    path.write_text(json.dumps(brief), encoding="utf-8")
    return str(path)


# ---------- reading the brief ----------

def test_missing_brief_explains_what_to_do(tmp_path):
    with pytest.raises(BriefError, match="Interview brief for Agent 4"):
        load_brief(str(tmp_path / "nope.json"))


def test_unreadable_and_wrong_files_are_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(BriefError, match="Could not read"):
        load_brief(str(bad))
    with pytest.raises(BriefError, match="not an interview brief"):
        load_brief(write(tmp_path, {"hello": 1}))
    with pytest.raises(BriefError, match="not an interview brief"):
        load_brief(write(tmp_path, ["a list"]))


def test_unknown_version_and_missing_title_are_rejected(tmp_path):
    brief = make_brief()
    brief["brief_version"] = 2
    with pytest.raises(BriefError, match="not supported"):
        load_brief(write(tmp_path, brief))
    with pytest.raises(BriefError, match="no job title"):
        load_brief(write(tmp_path, make_brief(title="  ")))


def test_brief_older_than_the_latest_run_is_stale(tmp_path):
    (tmp_path / "run_trace.json").write_text(json.dumps({"started": "2026-11-18T11:00:00"}), encoding="utf-8")
    with pytest.raises(BriefError, match="earlier run"):
        load_brief(write(tmp_path, make_brief()))          # created 10:00 < started 11:00


def test_brief_from_this_run_is_accepted(tmp_path):
    (tmp_path / "run_trace.json").write_text(json.dumps({"started": "2026-11-18T09:59:00"}), encoding="utf-8")
    assert load_brief(write(tmp_path, make_brief()))["job"]["title"] == "Junior Java Developer"


def test_no_trace_or_unreadable_trace_skips_the_stale_check(tmp_path):
    assert load_brief(write(tmp_path, make_brief()))
    (tmp_path / "run_trace.json").write_text("garbage", encoding="utf-8")
    assert load_brief(write(tmp_path, make_brief()))


# ---------- the questions ----------

def test_question_count_and_frame():
    questions = build_questions(make_brief())
    assert 4 <= len(questions) <= 5
    assert [q["id"] for q in questions] == list(range(1, len(questions) + 1))
    assert questions[0]["kind"] == "intro" and "Junior Java Developer" in questions[0]["question"]
    assert "Maplewood Software" in questions[0]["question"]
    assert questions[-1]["kind"] == "closing"


def test_gaps_come_before_claims_and_use_the_topic():
    questions = build_questions(make_brief())
    kinds = [q["kind"] for q in questions]
    assert kinds.index("gap") < kinds.index("claim")
    gap = next(q for q in questions if q["kind"] == "gap")
    assert gap["topic"] == "Linux" and "Linux" in gap["question"] and "don't see it on your resume" in gap["question"]
    claim = next(q for q in questions if q["kind"] == "claim")
    assert claim["topic"] == "Java" and "Java" in claim["question"]
    assert "(" not in gap["question"].split("Linux")[0]      # the note in brackets is not copied into the text


def test_a_long_brief_is_capped_at_five_questions():
    brief = make_brief(interview_focus=[f"Skill{i} (asked for, not on the resume)" for i in range(6)],
                       gaps=[f"Skill{i}" for i in range(6)])
    questions = build_questions(brief)
    assert len(questions) == 5
    assert len({q["topic"] for q in questions if q["topic"]}) == len([q for q in questions if q["topic"]])


def test_project_question_when_there_is_room():
    questions = build_questions(make_brief())          # 2 focus items + the project = 3 content questions
    project = next(q for q in questions if q["kind"] == "project")
    assert "Car Rental System" in project["question"] and "Lead Developer" in project["question"]


def test_gaps_are_used_when_interview_focus_is_empty():
    questions = build_questions(make_brief(interview_focus=[], gaps=["Docker", "5+ years of experience (candidate has 2.6)"]))
    kinds = [q["kind"] for q in questions]
    assert "gap" in kinds and "experience" in kinds
    experience = next(q for q in questions if q["kind"] == "experience")
    assert "5+ years" in experience["question"] and "2.6" in experience["question"]


def test_bare_skill_names_from_the_rule_based_fallback():
    questions = build_questions(make_brief(interview_focus=["Linux", "Java"], gaps=["Linux"]))
    by_topic = {q["topic"]: q["kind"] for q in questions if q["topic"]}
    assert by_topic["Linux"] == "gap" and by_topic["Java"] == "claim"


def test_general_fit_placeholder_is_ignored_and_generic_questions_fill_in():
    brief = make_brief(interview_focus=["General fit for the Data Analyst role"], gaps=[])
    brief["candidate"]["projects"] = []
    questions = build_questions(brief)
    assert 4 <= len(questions) <= 5
    assert [q["kind"] for q in questions].count("general") == 2
    assert not any("General fit" in q["question"] for q in questions)


def test_custom_role_with_no_company_or_evidence():
    brief = make_brief(title="Data Engineer", company=None, interview_focus=[], gaps=[])
    brief["candidate"] = {}
    questions = build_questions(brief)
    assert 4 <= len(questions) <= 5
    assert " at None" not in questions[0]["question"] and "Data Engineer" in questions[0]["question"]


def test_templates_only_no_llm(monkeypatch):
    import llm
    monkeypatch.setattr(llm, "ask_llm", lambda *a, **k: pytest.fail("Agent 4 must not call an LLM"))
    build_questions(make_brief())


def test_every_question_has_the_fields_the_next_steps_need():
    for q in build_questions(make_brief()):
        assert set(q) == {"id", "kind", "topic", "source", "question"}
        assert q["question"].strip()


# ---------- build_interview (the questions part of run_interview) ----------

def test_build_interview_returns_job_and_questions(tmp_path):
    result = interview_agent.build_interview(write(tmp_path, make_brief()))
    assert result["job"] == {"id": "job-1", "title": "Junior Java Developer", "company": "Maplewood Software"}
    assert 4 <= len(result["questions"]) <= 5


def test_build_interview_without_a_brief_raises(tmp_path):
    with pytest.raises(BriefError):
        interview_agent.build_interview(str(tmp_path / "missing.json"))


def test_command_line_prints_questions_or_a_clear_error(tmp_path, capsys):
    assert interview_agent.main(["--brief", write(tmp_path, make_brief()), "--questions-only"]) == 0
    assert "Q1 [intro]" in capsys.readouterr().out
    assert interview_agent.main(["--brief", str(tmp_path / "missing.json")]) == 1
    assert "Agent 4 cannot start" in capsys.readouterr().err


def test_years_question_takes_the_candidate_years_from_the_gaps():
    brief = make_brief(interview_focus=["3+ years of experience (asked for, not on the resume)"],
                       gaps=["3+ years of experience (candidate has 1.0)"])
    experience = next(q for q in build_questions(brief) if q["kind"] == "experience")
    assert "3+ years" in experience["question"] and "about 1.0" in experience["question"]


# ---------- degree gaps get an education question ----------

def test_degree_in_gaps_asks_about_education():
    brief = make_brief(interview_focus=[], gaps=["Bachelor's degree"])
    gap = next(q for q in build_questions(brief) if q["kind"] == "gap")
    assert gap["topic"] == "Bachelor's degree"
    assert gap["question"] == ("The posting lists a Bachelor's degree. Tell me about your education "
                               "and how it prepares you for this role.")
    assert "get up to speed" not in gap["question"]
    assert set(gap) == {"id", "kind", "topic", "source", "question"}   # same output format


def test_degree_in_interview_focus_asks_about_education():
    brief = make_brief(interview_focus=["Bachelor's degree (asked for, not on the resume)"],
                       gaps=["Bachelor's degree"])
    gap = next(q for q in build_questions(brief) if q["kind"] == "gap")
    assert gap["source"] == "interview_focus"
    assert gap["question"].startswith("The posting lists a Bachelor's degree.")
    assert "education" in gap["question"] and "get up to speed" not in gap["question"]


@pytest.mark.parametrize("topic, article", [("Master's degree", "a"), ("PhD", "a"),
                                            ("Associate's degree", "an"), ("doctorate", "a")])
def test_other_degrees_ask_about_education(topic, article):
    brief = make_brief(interview_focus=[], gaps=[topic])
    gap = next(q for q in build_questions(brief) if q["kind"] == "gap")
    assert gap["question"].startswith(f"The posting lists {article} {topic}.")


def test_skill_gap_keeps_the_old_wording():
    brief = make_brief(interview_focus=["Linux (asked for, not on the resume)"], gaps=["Linux"])
    gap = next(q for q in build_questions(brief) if q["kind"] == "gap")
    assert gap["question"] == ("The posting asks for Linux, but I don't see it on your resume. What "
                               "experience do you have with it, and how would you get up to speed quickly?")


# ---------- "your resume shows" only for skills and projects on the resume ----------

UNGROUNDED = ("authoritative Best Colleges (ask for an example; resume: Rank: 1/39 "
              "GPA:Overall3.95/4.3;Major4.05/4.3 Teaching: TeachingassistantofCalculus)")


def test_claim_backed_only_by_a_free_text_resume_line_is_skipped():
    brief = make_brief(interview_focus=[UNGROUNDED, "Java (ask for an example; resume: skills: Java)"],
                       gaps=[])
    questions = build_questions(brief)
    assert all("Best Colleges" not in (q["topic"] or "") for q in questions)
    assert all("Best Colleges" not in q["question"] for q in questions)
    claims = [q for q in questions if q["kind"] == "claim"]
    assert [q["topic"] for q in claims] == ["Java"]


def test_skipped_claim_does_not_use_up_a_question():
    # 1 ungrounded claim + 1 real claim + the project: all 3 slots still go to real topics
    brief = make_brief(interview_focus=[UNGROUNDED, "Java (ask for an example; resume: skills: Java)"],
                       gaps=[])
    content = [q for q in build_questions(brief) if q["kind"] not in ("intro", "closing")]
    assert [(q["kind"], q["topic"]) for q in content] == [("claim", "Java"),
                                                          ("project", "Car Rental System")]


def test_only_ungrounded_claims_fall_back_to_generic_questions():
    brief = make_brief(interview_focus=[UNGROUNDED], gaps=[])
    brief["candidate"]["projects"] = []
    content = [q for q in build_questions(brief) if q["kind"] not in ("intro", "closing")]
    assert [q["kind"] for q in content] == ["general", "general"]


def test_skill_claim_with_skills_evidence_still_asks_for_an_example():
    brief = make_brief(interview_focus=["Java (ask for an example; resume: skills: Java)"], gaps=[])
    claim = next(q for q in build_questions(brief) if q["kind"] == "claim")
    assert claim["topic"] == "Java"
    assert claim["question"].startswith("Your resume shows Java. Walk me through a specific example")


def test_skills_evidence_grounds_a_claim_even_if_not_in_the_candidate_skills():
    brief = make_brief(interview_focus=["Git (ask for an example; resume: skills: Git)"], gaps=[])
    claim = next(q for q in build_questions(brief) if q["kind"] == "claim")
    assert claim["topic"] == "Git"


def test_bare_skill_on_the_resume_still_asks_for_an_example():
    brief = make_brief(interview_focus=["java"], gaps=[])          # rule-based fallback, any case
    claim = next(q for q in build_questions(brief) if q["kind"] == "claim")
    assert claim["question"].startswith("Your resume shows java.")


def test_a_project_name_counts_as_on_the_resume():
    brief = make_brief(interview_focus=["car rental system"], gaps=[])
    brief["candidate"]["skills"] = []
    assert [q["topic"] for q in build_questions(brief) if q["kind"] == "claim"] == ["car rental system"]


def test_a_skill_listed_under_a_project_counts_as_on_the_resume():
    brief = make_brief(interview_focus=["OOP"], gaps=[])
    assert not [q for q in build_questions(brief) if q["kind"] == "claim"]   # not on this resume
    brief["candidate"]["projects"][0]["skills"] = ["Java", "OOP"]
    assert [q["topic"] for q in build_questions(brief) if q["kind"] == "claim"] == ["OOP"]


def test_bare_phrase_not_on_the_resume_is_skipped():
    brief = make_brief(interview_focus=["stakeholder communication"], gaps=[])
    assert not [q for q in build_questions(brief) if q["kind"] == "claim"]


def test_gap_items_are_unchanged():
    brief = make_brief(interview_focus=["Kubernetes (asked for, not on the resume)"], gaps=["Kubernetes"])
    gap = next(q for q in build_questions(brief) if q["kind"] == "gap")
    assert gap["topic"] == "Kubernetes" and gap["source"] == "interview_focus"
    assert gap["question"].startswith("The posting asks for Kubernetes, but I don't see it on your resume.")
