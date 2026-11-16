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
    embedding_assessment, extract_requirements, nice_to_have_skills, rank_jobs, score_match,
)
from keypoints import COVERED_AT, candidate_phrases, extract_keyphrases, match_phrases, resume_lines

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

def test_resume_lines_skip_headings():
    assert resume_lines("Skills\nWork Experience\nBuilt a chess engine in Java") == \
        ["Built a chess engine in Java"]


def test_match_by_skill_by_similarity_and_not_at_all(fake_model):
    lines = resume_lines(STUDENT["raw_text"])
    matches = match_phrases([("C++", 0.5), ("internal tools", 0.4), ("Tableau dashboards", 0.3)],
                            lines, STUDENT["skills"])
    by_phrase = {m["phrase"]: m for m in matches}
    assert by_phrase["C++"]["covered"] and by_phrase["C++"]["evidence"] == "skills: C++"
    tools = by_phrase["internal tools"]
    assert tools["covered"] and tools["resume_similarity"] >= COVERED_AT
    assert tools["evidence"] == "Built internal tools for the campus library in Java."
    assert not by_phrase["Tableau dashboards"]["covered"]
    assert by_phrase["Tableau dashboards"]["evidence"] is None


def test_match_with_empty_resume(fake_model):
    matches = match_phrases([("Java", 0.5)], [], [])
    assert matches[0]["covered"] is False and matches[0]["resume_similarity"] == 0.0


# ---------- the assessment built from the matches ----------

def test_nice_to_have_from_plus_bonus_preferred():
    assert nice_to_have_skills(JOBS["job-001"]["description"], ["Java", "C++", "Git"]) == ["C++"]
    assert nice_to_have_skills(JOBS["job-002"]["description"], ["C++", "Git", "GitHub"]) == ["Git", "GitHub"]
    assert nice_to_have_skills(JOBS["job-003"]["description"], ["Python", "Tableau"]) == ["Tableau"]


def test_assessment_fields_and_interview_order():
    job = JOBS["job-001"]
    req = extract_requirements(job)
    b = score_match(STUDENT, job)["breakdown"]
    matches = [{"phrase": "internal tools", "relevance": 0.6, "resume_similarity": 0.7,
                "covered": True, "evidence": "Built internal tools in Java."},
               {"phrase": "clean code", "relevance": 0.3, "resume_similarity": 0.2,
                "covered": False, "evidence": None}]
    a = embedding_assessment(job, b, req, matches)
    assert a["key_points"] == ["internal tools", "clean code"]
    assert "C++" in a["nice_to_have"] and "C++" not in a["required_skills"]
    assert "clean code" in a["gaps"]
    assert a["interview_focus"][0].endswith("(asked for, not on the resume)")    # gaps first
    assert a["interview_focus"][-1].startswith("internal tools (ask for an example")
    assert "1 of 2 key points" in a["fit_summary"]
    assert a["key_point_matches"] == matches


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


def test_model_failure_warns_once_and_uses_rules(monkeypatch, capsys):
    ranked = rank_jobs(STUDENT, list(JOBS.values()))          # conftest blocks KeyBERT
    assert {r["assessed_by"] for r in ranked} == {"rules"}
    assert capsys.readouterr().err.count("Embedding key points unavailable") == 1
