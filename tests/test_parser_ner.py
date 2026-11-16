"""Tests for the NER version of Agent 1 (parse_resume_ner).

Most of these tests use a FAKE model that
returns hand-checked annotations (tests/fixtures/ner/handmade_eval.jsonl),
i.e. a "perfect" NER model. That tests how Python assembles entities into
the profile, separately from how good the model is (scripts/train_ner.py
measures that). The last tests load the real committed model and fail if
it is missing, broken, or scores F1 < 0.90 on the hand-annotated resumes.
"""
import json
import os
import sys

import pytest

import resume_parser
from resume_parser import (
    NER_CONFIG, NERUnavailable, build_profile_from_ner, ner_entities, non_job_lines,
    parse_date, parse_resume, parse_resume_ner,
)
from pipeline import run_pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))             # for train_ner.evaluate
FIXTURES = os.path.join(ROOT, "tests", "fixtures")
SKILLS = os.path.join(ROOT, "data", "skills.txt")
with open(os.path.join(FIXTURES, "ner", "handmade_eval.jsonl"), encoding="utf-8") as f:
    GOLD = {json.loads(line)["id"]: json.loads(line) for line in f}
with open(os.path.join(FIXTURES, "jobs.json"), encoding="utf-8") as f:
    JOBS = json.load(f)


class FakeEnt:
    def __init__(self, text, start, end, label):
        self.start_char, self.end_char, self.label_ = start, end, label
        self.text = text[start:end]


class FakeDoc:
    def __init__(self, ents):
        self.ents = ents


class FakeNER:
    """Returns given entities for a text: the gold annotations by default,
    or any (start, end, label) list a test wants to try."""
    def __init__(self, extra=None, drop=None):
        self.extra, self.drop = extra or [], drop or set()

    def __call__(self, text):
        gold = next((r["entities"] for r in GOLD.values() if r["text"] == text), [])
        spans = [(s, e, label) for s, e, label, t in gold if t not in self.drop] + self.extra
        return FakeDoc([FakeEnt(text, *span) for span in sorted(spans)])


@pytest.fixture
def fake_ner(monkeypatch):
    def install(model=None):
        model = model or FakeNER()
        monkeypatch.setattr(resume_parser, "_load_ner_model", lambda: model)
        return model
    return install


def path(name):
    return os.path.join(FIXTURES, name)


# ---------- a perfect model gives the same profile as the rules ----------

@pytest.mark.parametrize("name", ["resume_title_case.docx", "resume_numeric_dates.docx",
                                  "resume_comma_skills.docx"])
def test_perfect_model_matches_rule_based_profile(fake_ner, name):
    fake_ner()
    ner, rules = parse_resume_ner(path(name), SKILLS), parse_resume(path(name), SKILLS)
    assert ner["parsed_by"] == "ner"
    assert list(ner) == list(rules)                              # same keys, same order
    for key in ["name", "email", "phone", "linkedin", "github", "education",
                "experience", "experience_years", "projects", "raw_text"]:
        assert ner[key] == rules[key], key
    assert sorted(ner["skills"]) == sorted(rules["skills"])


def test_jobs_and_date_math(fake_ner):
    fake_ner()
    jobs = parse_resume_ner(path("resume_numeric_dates.docx"), SKILLS)["experience"]
    assert jobs == [
        {"company": "Lakeside Robotics", "title": "Software Engineer Intern",
         "dates": "01/2023 - 05/2024", "months": 17},           # Python counts the months
        {"company": "Harbor Tech Repair", "title": "IT Support Technician",
         "dates": "06/2021 - 12/2022", "months": 19},
    ]


def test_hyphenated_name_kept_whole(fake_ner):
    fake_ner()
    assert parse_resume_ner(path("resume_comma_skills.docx"), SKILLS)["name"] == "Elena Vasquez-Moreau"


# ---------- realistic model mistakes are filtered ----------

def test_fake_dates_are_ignored(fake_ner):
    # The model tagging "weekly" as a DATE (it really did) must not break the dates
    text = GOLD["resume_title_case.docx"]["text"]
    start = text.index("weekly")
    fake_ner(FakeNER(extra=[(start, start + len("weekly"), "DATE")]))
    jobs = parse_resume_ner(path("resume_title_case.docx"), SKILLS)["experience"]
    assert [j["months"] for j in jobs] == [24, 16]
    assert parse_date("weekly") is None and parse_date("three years") is None


def test_project_and_leadership_titles_are_not_jobs(fake_ner):
    # The model tagging a project role as JOB_TITLE (it really did) must not add a job
    text = GOLD["resume_title_case.docx"]["text"]
    start = text.index("Team Lead")
    fake_ner(FakeNER(extra=[(start, start + len("Team Lead"), "JOB_TITLE")]))
    jobs = parse_resume_ner(path("resume_title_case.docx"), SKILLS)["experience"]
    assert [j["title"] for j in jobs] == ["Junior Data Analyst", "Student Assistant"]


def test_non_job_sections():
    text = "Jane\nEXPERIENCE\nAcme | Dev\nPROJECTS\nBot | Lead\nLEADERSHIP\nTreasurer | Club\nSKILLS\nPython"
    assert non_job_lines(text) == {4, 6}


def test_skills_in_links_are_dropped(fake_ner):
    # "GitHub" as a link label and "github" inside the link are tagged SKILL by
    # mistake; only the real "Python" should survive
    text = "Ann Lee\nGitHub: github.com/annlee\nSkills\nPython"
    py = text.index("Python")
    ents = [FakeEnt(text, 0, 7, "PERSON"), FakeEnt(text, 8, 14, "SKILL"),
            FakeEnt(text, 16, 22, "SKILL"), FakeEnt(text, py, py + 6, "SKILL")]
    profile = build_profile_from_ner(ner_entities(lambda t: FakeDoc(ents), text), text, ["Python", "GitHub"])
    assert profile["skills"] == ["Python"]


def test_skill_spelling_from_skills_list(fake_ner):
    text = "Ann Lee\nSkills\npython, Kubernetes"
    ents = [FakeEnt(text, 0, 7, "PERSON"), FakeEnt(text, 15, 21, "SKILL"), FakeEnt(text, 23, 33, "SKILL")]
    profile = build_profile_from_ner(ner_entities(lambda t: FakeDoc(ents), text), text, ["Python"])
    assert profile["skills"] == ["Python", "Kubernetes"]     # new skills beyond the list are kept


def test_all_caps_name_is_title_cased():
    text = "ANN LEE\nann@example.com"
    ents = [FakeEnt(text, 0, 7, "PERSON")]
    profile = build_profile_from_ner(ner_entities(lambda t: FakeDoc(ents), text), text, [])
    assert profile["name"] == "Ann Lee"


def test_job_without_dates_has_zero_months():
    text = "Ann Lee\nEXPERIENCE\nAcme Corp | Developer"
    ents = [FakeEnt(text, 0, 7, "PERSON"), FakeEnt(text, 19, 28, "ORG"), FakeEnt(text, 31, 40, "JOB_TITLE")]
    profile = build_profile_from_ner(ner_entities(lambda t: FakeDoc(ents), text), text, [])
    assert profile["experience"] == [{"company": "Acme Corp", "title": "Developer", "dates": None, "months": 0}]


def test_entity_line_numbers():
    text = "Ann Lee\nAcme Corp"
    ents = ner_entities(lambda t: FakeDoc([FakeEnt(text, 0, 7, "PERSON"), FakeEnt(text, 8, 17, "ORG")]), text)
    assert [(e["label"], e["line"]) for e in ents] == [("PERSON", 0), ("ORG", 1)]


# ---------- fallback ----------

def test_falls_back_to_rules_when_model_missing(capsys):
    # conftest.py blocks the real model: that's the "model missing" case
    profile = parse_resume_ner(path("resume_title_case.docx"), SKILLS)
    assert profile["parsed_by"] == "rules"
    assert profile == parse_resume(path("resume_title_case.docx"), SKILLS)
    warning = capsys.readouterr().err                         # the fallback is loud
    assert "WARNING [Agent 1]" in warning
    assert "disabled in tests" in warning                     # says why
    assert "RULE-BASED parser" in warning


def test_falls_back_when_model_crashes(fake_ner, capsys):
    def broken(text):
        raise RuntimeError("corrupt model")
    fake_ner(broken)
    assert parse_resume_ner(path("resume_title_case.docx"), SKILLS)["parsed_by"] == "rules"
    warning = capsys.readouterr().err
    assert "crashed while reading the resume: RuntimeError: corrupt model" in warning


def test_loader_reports_missing_model(monkeypatch):
    monkeypatch.undo()                                        # use the real loader
    monkeypatch.setattr(resume_parser, "_ner_model", None)
    monkeypatch.setitem(NER_CONFIG, "model_path", os.path.join(ROOT, "no_such_model_dir"))
    with pytest.raises(NERUnavailable, match="model folder not found"):
        resume_parser._load_ner_model()


def test_loader_reports_broken_model(monkeypatch, tmp_path):
    # A folder that exists but isn't a spaCy model
    monkeypatch.undo()
    monkeypatch.setattr(resume_parser, "_ner_model", None)
    monkeypatch.setitem(NER_CONFIG, "model_path", str(tmp_path))
    with pytest.raises(NERUnavailable, match="spaCy could not load"):
        resume_parser._load_ner_model()


# ---------- pipeline default ----------

def test_pipeline_uses_ner_by_default(fake_ner):
    fake_ner()
    result = run_pipeline(path("resume_title_case.docx"), JOBS)
    assert result["profile"]["parsed_by"] == "ner"
    assert "raw_text" not in result["profile"]


def test_pipeline_rejects_unknown_parser():
    with pytest.raises(ValueError):
        run_pipeline(path("resume_title_case.docx"), JOBS, parser="magic")


# ---------- the real, committed model (fails loudly if it breaks) ----------

@pytest.fixture
def real_model(monkeypatch):
    monkeypatch.undo()                                        # allow the real model
    monkeypatch.setattr(resume_parser, "_ner_model", None)    # load it fresh from disk
    return resume_parser._load_ner_model()


def test_committed_model_f1_on_handmade_set(real_model):
    # Loads models/resume_ner/ the same way the app does and scores it on the
    # 3 hand-annotated resumes. Fails if the model is missing, corrupt, or worse.
    from train_ner import evaluate, load_records
    scores = evaluate(real_model, load_records("handmade_eval.jsonl"))
    assert scores["overall"]["f1"] >= 0.90, scores["overall"]


def test_real_model_on_fixture(real_model, capsys):
    profile = parse_resume_ner(path("resume_numeric_dates.docx"), SKILLS)
    assert profile["parsed_by"] == "ner"
    assert profile["name"] == "Marcus Oyelaran"
    assert profile["experience_years"] == 3.0
    assert "WARNING" not in capsys.readouterr().err
