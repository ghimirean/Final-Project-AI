"""Tests for Agent 3's embedding key points (agents/keypoints.py + the
embedding_assessment in job_synthesizer.py).

The REAL KeyBERT algorithm runs here, but on a FAKE embedding model (no
download, no sentence-transformers model). Candidate phrases come from the
real spaCy en_core_web_sm model, which is installed by requirements.txt.
"""
import json
import os

import numpy as np
import pytest

import embeddings
import keypoints
import llm
from job_synthesizer import (
    SKILLS_PATH, coverage, embedding_assessment, extract_requirements, load_skills,
    nice_to_have_skills, phrase_kind, rank_jobs, score_match,
)
from keypoints import (
    COVERED_AT, candidate_phrases, extract_keyphrases, is_evidence, match_phrases, resume_lines,
)

REAL_LOAD_KEYBERT = keypoints._load_keybert          # saved before conftest blocks it

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "tests", "fixtures", "jobs.json"), encoding="utf-8") as f:
    JOBS = {job["id"]: job for job in json.load(f)}

STUDENT = {
    "skills": ["Java", "C++", "Git", "OOP"],
    "education": {"degree": "Bachelor of Science in Computer Science"},
    "experience": [],
    "experience_years": 1.0,
    "raw_text": "Skills\nJava, C++, Git, OOP\nBuilt internal tools for the campus library in Java.\n"
                "Teaching assistant for data structures and algorithms.",
}

# The fake model "understands" only these words: one vector slot per word
VOCAB = ["java", "c++", "internal", "tools", "git", "code", "data", "structures",
         "python", "pandas", "sql", "tableau", "dashboards", "developer", "junior"]


class FakeModel:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, normalize_embeddings=True):
        self.calls += 1
        vectors = np.array([[t.lower().count(w) for w in VOCAB] for t in texts], dtype=float)
        lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(lengths == 0, 1, lengths)


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(embeddings, "_load_model", lambda: model)
    return model


@pytest.fixture
def keybert(fake_model, monkeypatch):
    """Real KeyBERT on top of the fake embedding model."""
    monkeypatch.setattr(keypoints, "_keybert", None)
    monkeypatch.setattr(keypoints, "_load_keybert", REAL_LOAD_KEYBERT)
    return fake_model


# ---------- candidate phrases (spaCy noun phrases) ----------

def test_candidates_are_readable_noun_phrases():
    phrases = candidate_phrases(JOBS["job-002"]["description"])
    assert phrases["performance critical c++ services"] == "performance-critical C++ services"
    assert phrases["summer internship"] == "Summer internship"


def test_filler_words_are_trimmed():
    shown = set(candidate_phrases(JOBS["job-001"]["description"]).values())
    assert "Requirements" not in shown                       # filler on its own is dropped
    assert "OOP principles" in shown and "internal tools" in shown


def test_no_spacy_returns_none(monkeypatch):
    def broken():
        raise OSError("model not installed")
    monkeypatch.setattr(keypoints, "_load_spacy", broken)
    assert candidate_phrases("anything") is None


# ---------- KeyBERT key phrases ----------

def test_keyphrases_come_from_the_posting(keybert):
    job = JOBS["job-001"]["description"]
    phrases = extract_keyphrases(job, top_n=4)
    shown = set(candidate_phrases(job).values())
    assert 1 <= len(phrases) <= 4
    assert len({p for p, _ in phrases}) == len(phrases)      # no duplicates
    assert all(p in shown for p, _ in phrases)               # readable original phrases
    assert all(0 <= r <= 1 for _, r in phrases)
    # The fake model only "knows" VOCAB words, so the best phrase must use one
    assert any(w in phrases[0][0].lower() for w in VOCAB)


def test_keyphrases_without_spacy_still_work(keybert, monkeypatch):
    monkeypatch.setattr(keypoints, "_load_spacy", lambda: (_ for _ in ()).throw(OSError("no")))
    phrases = extract_keyphrases(JOBS["job-001"]["description"], top_n=3)
    assert phrases and all(isinstance(p, str) for p, _ in phrases)


def test_keybert_blocked_raises_embedding_error():
    # conftest.py blocks KeyBERT: the caller must see an EmbeddingError
    with pytest.raises(embeddings.EmbeddingError):
        extract_keyphrases(JOBS["job-001"]["description"])


# ---------- matching key phrases against the resume ----------

def test_resume_lines_skip_headings_and_join_wrapped_lines():
    text = ("Skills\nWork Experience\n• Built a chess engine in Java\n"
            "• Answered tickets for the help desk\nand multi-customer support")
    assert resume_lines(text) == [
        "Built a chess engine in Java",                                      # bullet removed
        "Answered tickets for the help desk and multi-customer support",    # wrapped line joined
    ]


def test_match_finds_the_best_line(fake_model):
    lines = resume_lines(STUDENT["raw_text"])
    by_phrase = {m["phrase"]: m for m in match_phrases(
        [("internal tools", 0.4), ("Tableau dashboards", 0.3)], lines)}
    assert by_phrase["internal tools"]["best_line"] == "Built internal tools for the campus library in Java."
    assert by_phrase["internal tools"]["resume_similarity"] >= COVERED_AT
    assert by_phrase["Tableau dashboards"]["resume_similarity"] < COVERED_AT


def test_match_with_empty_resume(fake_model):
    m = match_phrases([("Java", 0.5)], [])[0]
    assert m["resume_similarity"] == 0.0 and m["best_line"] is None


def test_evidence_rules():
    line = "Answered tickets for the help desk and multi-customer support"
    assert not is_evidence("customer data", line, 0.50)          # shares one word only: needs 0.55
    assert is_evidence("customer data", line, 0.56)
    assert is_evidence("building websites", "Built client sites with React", 0.47)   # no shared word is fine
    assert not is_evidence("building websites", "Built client sites with React", 0.44)
    assert is_evidence("dashboards", "Built weekly sales dashboards in Tableau", 0.46)  # one-word phrase


# ---------- coverage: skills and degrees by rules, the rest by evidence ----------

DESC = JOBS["job-001"]["description"]     # "... degree in Computer Science or a related field ..."
SKILLS = load_skills(SKILLS_PATH)


def cover(phrase, kind, best_line=None, similarity=0.0, have=(), degree="", description=DESC):
    m = {"phrase": phrase, "resume_similarity": similarity, "best_line": best_line}
    return coverage(m, kind, description, {s.lower() for s in have}, {"degree": degree}, SKILLS)


def test_skill_phrases_need_the_exact_skill():
    assert cover("GitHub", "skill", "Set up Git for the team", 0.6, have=["Git"]) == (False, None)
    assert cover("performance-critical C++ services", "skill", have=["C++"]) == (True, "skills: C++")


def test_degree_level_and_field():
    bs_ce = "Bachelor of Science in Computer Engineering"
    assert cover("Bachelor", "degree", degree=bs_ce) == (True, bs_ce)
    assert cover("Bachelor", "degree", degree="Associate of Science in Computer Science")[0] is False
    assert cover("Computer Science", "degree", degree="B.S. in Computer Science")[0] is True
    senior = JOBS["job-005"]["description"]                       # no "related field" here
    assert cover("Computer Science", "degree", degree=bs_ce, description=senior) == (False, None)
    covered, evidence = cover("Computer Science", "degree", degree=bs_ce)   # job-001 accepts related
    assert covered and "related field" in evidence


def test_experience_and_role_use_the_evidence_rules():
    line = "Built client sites with React and Node.js."
    assert cover("building websites", "experience", line, 0.47) == (True, line)
    assert cover("customer data", "role", "Answered tickets and multi-customer support", 0.5) == (False, None)


# ---------- the assessment built from the matches ----------

def test_nice_to_have_from_plus_bonus_preferred():
    assert nice_to_have_skills(JOBS["job-001"]["description"], ["Java", "C++", "Git"]) == ["C++"]
    assert nice_to_have_skills(JOBS["job-002"]["description"], ["C++", "Git", "GitHub"]) == ["Git", "GitHub"]
    assert nice_to_have_skills(JOBS["job-003"]["description"], ["Python", "Tableau"]) == ["Tableau"]


def match(phrase, best_line=None, similarity=0.2):
    # what keypoints.match_phrases returns for one phrase
    return {"phrase": phrase, "relevance": 0.5, "resume_similarity": similarity,
            "best_line": best_line}


@pytest.mark.parametrize("job_id, phrase, kind", [
    ("job-002", "performance-critical C++ services", "skill"),   # contains a listed skill
    ("job-001", "Bachelor", "degree"),                           # is a degree
    ("job-001", "Computer Science", "degree"),                   # "degree in Computer Science"
    ("job-002", "Computer Science", "degree"),                   # "students studying Computer Science"
    ("job-005", "professional backend", "experience"),           # "5+ years of professional backend ..."
    ("job-006", "building websites", "experience"),              # "1 year of experience building websites"
    ("job-005", "Computer Science", "degree"),                   # same sentence, after the degree word
    ("job-001", "junior developer", "role"),
    ("job-002", "Summer internship", "role"),
    ("job-006", "designers", "role"),
    ("job-006", "small business clients", "role"),
])
def test_phrase_kind(job_id, phrase, kind):
    assert phrase_kind(phrase, JOBS[job_id]["description"], load_skills(SKILLS_PATH)) == kind


def test_assessment_fields_and_interview_order():
    job = JOBS["job-001"]
    req = extract_requirements(job)
    b = score_match(STUDENT, job)["breakdown"]
    matches = [match("internal tools", "Built internal tools in Java.", 0.6),
               match("Computer Science")]
    a = embedding_assessment(job, b, req, matches)            # no profile: no degree on file
    assert a["key_points"] == ["internal tools", "Computer Science"]
    assert "C++" in a["nice_to_have"] and "C++" not in a["required_skills"]
    assert "Computer Science" in a["gaps"]                    # a degree requirement
    assert a["interview_focus"][0].endswith("(asked for, not on the resume)")    # gaps first
    assert a["interview_focus"][-1].startswith("internal tools (ask for an example")
    assert "0 of the 1 requirements" in a["fit_summary"]      # role phrases aren't requirements
    assert [m["kind"] for m in a["key_point_matches"]] == ["role", "degree"]


def test_role_descriptions_are_never_gaps():
    job = JOBS["job-006"]
    b = score_match(STUDENT, job)["breakdown"]
    matches = [match("designers"), match("small business clients"), match("building websites")]
    a = embedding_assessment(job, b, extract_requirements(job), matches)
    assert a["key_points"] == ["designers", "small business clients", "building websites"]
    assert "designers" not in a["gaps"] and "small business clients" not in a["gaps"]
    assert "building websites" in a["gaps"]                   # an experience requirement
    probes = " ".join(a["interview_focus"])
    assert "designers" not in probes and "small business clients" not in probes


# ---------- rank_jobs: embeddings by default, no LLM ----------

def test_rank_jobs_uses_embeddings_and_never_calls_groq(keybert, monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "_post", lambda *a: calls.append(a))
    ranked = rank_jobs(STUDENT, list(JOBS.values()))
    assert calls == []                                        # no LLM in the default path
    assert {r["assessed_by"] for r in ranked} == {"embeddings"}
    for r in ranked:
        a = r["assessment"]
        assert a["key_points"] and a["interview_focus"]
        assert len(a["key_point_matches"]) == len(a["key_points"])


def test_key_points_never_change_the_score(keybert, monkeypatch):
    with_key_points = [(r["id"], r["score"]) for r in rank_jobs(STUDENT, list(JOBS.values()))]
    monkeypatch.setattr(keypoints, "_load_keybert", lambda: (_ for _ in ()).throw(
        embeddings.EmbeddingError("off")))
    rules_only = [(r["id"], r["score"]) for r in rank_jobs(STUDENT, list(JOBS.values()))]
    assert with_key_points == rules_only


def test_groq_failure_falls_back_to_embeddings(keybert, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")   # conftest makes the call fail
    ranked = rank_jobs(STUDENT, [JOBS["job-001"]], use_llm=True)
    assert ranked[0]["assessed_by"] == "embeddings"


def test_comparison_script_matches_llm_points_to_phrases(fake_model):
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from compare_key_points import captured
    pairs = captured(["Write clean Java code for internal tools", "Build Tableau dashboards"],
                     ["internal tools", "Tableau", "Git"])
    assert [phrase for phrase, _ in pairs] == ["internal tools", "Tableau"]
    assert all(0 <= s <= 1 for _, s in pairs)
    assert captured([], ["Java"]) == []


def test_model_failure_warns_once_and_uses_rules(monkeypatch, capsys):
    ranked = rank_jobs(STUDENT, list(JOBS.values()))          # conftest blocks KeyBERT
    assert {r["assessed_by"] for r in ranked} == {"rules"}
    assert capsys.readouterr().err.count("Embedding key points unavailable") == 1
