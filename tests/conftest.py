"""Shared test setup.

Safety nets, applied to EVERY test:
- No real LLM API calls: the one network function in agents/llm.py is
  replaced with one that always fails. Tests that need an LLM reply
  install their own fake (see tests/test_llm.py), which overrides it.
- No real embedding model (no downloads, no slow model loading): the one
  loader in agents/embeddings.py always fails, so Agent 3 uses its TF-IDF
  fallback unless a test installs a fake model (tests/test_embeddings.py).
- No KeyBERT: Agent 3's key points use the rule-based fallback unless a
  test turns KeyBERT on with a fake embedding model (tests/test_keypoints.py).
- No Whisper model and no microphone: the loaders in agents/interview_speech.py
  always fail, so Agent 4's speech tests install fakes (tests/test_interview_speech.py).
- No trained NER model by default: the loader in agents/resume_parser.py
  always fails, so Agent 1 uses its rule-based fallback unless a test
  installs a fake model (tests/test_parser_ner.py) or undoes this to load
  the real committed model.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agents"))

import embeddings  # noqa: E402
import interview_speech  # noqa: E402
import keypoints  # noqa: E402
import llm  # noqa: E402
import resume_parser  # noqa: E402


@pytest.fixture(autouse=True)
def no_real_llm_calls(monkeypatch):
    def blocked(url, headers, body):
        raise llm.LLMError("real network calls are disabled in tests")
    monkeypatch.setattr(llm, "_post", blocked)


@pytest.fixture(autouse=True)
def no_real_embedding_model(monkeypatch):
    def blocked():
        raise embeddings.EmbeddingError("the real embedding model is disabled in tests")
    monkeypatch.setattr(embeddings, "_load_model", blocked)


@pytest.fixture(autouse=True)
def no_real_keybert(monkeypatch):
    # KeyBERT imports sentence-transformers and torch (slow), so it is off by
    # default; tests/test_keypoints.py turns it on with a fake embedding model
    def blocked():
        raise embeddings.EmbeddingError("KeyBERT is disabled in tests")
    monkeypatch.setattr(keypoints, "_load_keybert", blocked)


@pytest.fixture(autouse=True)
def no_real_ner_model(monkeypatch):
    def blocked():
        raise resume_parser.NERUnavailable("the trained NER model is disabled in tests")
    monkeypatch.setattr(resume_parser, "_load_ner_model", blocked)


@pytest.fixture(autouse=True)
def no_real_whisper_or_microphone(monkeypatch):
    def blocked_model():
        raise interview_speech.SpeechError("the real Whisper model is disabled in tests")

    def blocked_sounddevice():
        raise interview_speech.SpeechError("the real microphone is disabled in tests")
    monkeypatch.setattr(interview_speech, "_load_whisper_model", blocked_model)
    monkeypatch.setattr(interview_speech, "_sounddevice", blocked_sounddevice)
