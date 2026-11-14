"""Tests for embeddings in Agent 3. The model is always faked: nothing is
downloaded and sentence-transformers is never loaded."""
import json
import os
import sys

import numpy as np
import pytest

import embeddings
from embeddings import EmbeddingError, embed, resume_job_similarities, split_chunks
from job_synthesizer import (
    EMBED_HIGH, EMBED_LOW, rank_jobs, score_match, similarity_to_points, text_scores,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "tests", "fixtures", "jobs.json"), encoding="utf-8") as f:
    JOBS = json.load(f)

STUDENT = {
    "skills": ["Java", "C++", "Git", "OOP", "Data Structures", "Algorithms"],
    "education": {"degree": "Bachelor of Science in Computer Science"},
    "experience": [],
    "experience_years": 1.0,
    "raw_text": "Computer science student. Built projects in Java and C++ using Git.",
}

# The fake model "understands" only these words: one vector slot per word
VOCAB = ["java", "c++", "python", "react", "sql", "docker", "git", "learning", "bakery"]


class FakeModel:
    """Stands in for SentenceTransformer: counts VOCAB words in each text."""
    def __init__(self):
        self.calls = []

    def encode(self, texts, normalize_embeddings=True):
        self.calls.append(list(texts))
        vectors = np.array([[t.lower().count(w) for w in VOCAB] for t in texts], dtype=float)
        lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(lengths == 0, 1, lengths)


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(embeddings, "_load_model", lambda: model)
    return model


# ---------- embeddings.py ----------

def test_split_chunks():
    assert len(split_chunks("word " * 250, size=120)) == 3
    assert split_chunks("") == [""]


def test_embed_returns_unit_vectors(fake_model):
    vectors = embed(["java and python", "bakery"])
    assert vectors.shape == (2, len(VOCAB))
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_similarity_uses_best_matching_chunk(fake_model):
    # First chunk is about a bakery, second chunk is about Java
    resume = "bakery " * 120 + "java " * 50
    java, bakery, nothing = resume_job_similarities(resume, ["java developer", "bakery", "nurse"])
    assert java == pytest.approx(1.0)
    assert bakery == pytest.approx(1.0)
    assert nothing == 0.0


def test_embedding_failure_raises_embedding_error(monkeypatch):
    class Broken:
        def encode(self, texts, normalize_embeddings=True):
            raise RuntimeError("out of memory")
    monkeypatch.setattr(embeddings, "_load_model", lambda: Broken())
    with pytest.raises(EmbeddingError):
        embed(["hello"])


# ---------- similarity -> points ----------

def test_similarity_to_points_mapping():
    assert similarity_to_points(EMBED_LOW) == 0
    assert similarity_to_points(EMBED_HIGH) == 40
    assert similarity_to_points((EMBED_LOW + EMBED_HIGH) / 2) == pytest.approx(20)
    assert similarity_to_points(-0.2) == 0          # never negative
    assert similarity_to_points(0.99) == 40         # never above the weight


# ---------- scoring with embeddings ----------

def test_scores_use_embeddings_when_model_available(fake_model):
    ranked = rank_jobs(STUDENT, JOBS)
    assert all(r["breakdown"]["text_method"] == "embeddings" for r in ranked)
    assert ranked[0]["id"] == "job-001"                       # Junior Java Developer


def test_score_does_not_depend_on_other_jobs(fake_model):
    # The point of absolute scoring: a job scores the same alone, in a short
    # list, or in the full list
    in_full_list = {r["id"]: r["score"] for r in rank_jobs(STUDENT, JOBS)}
    for job in JOBS:
        assert rank_jobs(STUDENT, [job])[0]["score"] == in_full_list[job["id"]]
        assert score_match(STUDENT, job)["score"] == in_full_list[job["id"]]


def test_all_jobs_embedded_in_one_call(fake_model):
    rank_jobs(STUDENT, JOBS)
    assert len(fake_model.calls) == 2                         # resume chunks, then all jobs
    assert len(fake_model.calls[1]) == len(JOBS)


# ---------- fallback to TF-IDF ----------

def test_falls_back_to_tfidf_when_model_unavailable():
    # conftest.py blocks the model loader, which is the "can't load" case
    ranked = rank_jobs(STUDENT, JOBS)
    assert all(r["breakdown"]["text_method"] == "tfidf" for r in ranked)
    assert max(r["breakdown"]["text_points"] for r in ranked) == 40.0   # relative scaling


def test_falls_back_when_embedding_fails_midway(monkeypatch):
    class Broken:
        def encode(self, texts, normalize_embeddings=True):
            raise RuntimeError("boom")
    monkeypatch.setattr(embeddings, "_load_model", lambda: Broken())
    assert text_scores(STUDENT, JOBS)[0]["method"] == "tfidf"


def test_real_model_library_is_never_loaded_by_tests():
    assert "sentence_transformers" not in sys.modules
