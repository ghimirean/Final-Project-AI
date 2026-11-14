"""Shared test setup.

Safety net: no test may ever call a real LLM API. This replaces the one
network function in agents/llm.py with one that always fails, for EVERY
test. Tests that need an LLM reply install their own fake (see
tests/test_llm.py), which overrides this one.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agents"))

import llm  # noqa: E402


@pytest.fixture(autouse=True)
def no_real_llm_calls(monkeypatch):
    def blocked(url, headers, body):
        raise llm.LLMError("real network calls are disabled in tests")
    monkeypatch.setattr(llm, "_post", blocked)
