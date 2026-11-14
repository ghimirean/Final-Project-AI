"""Tests for agents/llm.py. The LLM is always faked: no network calls."""
import pytest
from pydantic import BaseModel

import llm
from llm import ask_llm, parse_json, LLMError

FAKE_KEY = "test-key-not-real"

# Saved at import time, before conftest.py swaps _post out for each test
REAL_POST = llm._post


class Person(BaseModel):
    name: str
    age: int


def fake_gemini(reply_text, calls):
    def fake_post(url, headers, body):
        calls.append({"url": url, "headers": headers, "body": body})
        return {"candidates": [{"content": {"parts": [{"text": reply_text}]}}]}
    return fake_post


def fake_groq(reply_text, calls):
    def fake_post(url, headers, body):
        calls.append({"url": url, "headers": headers, "body": body})
        return {"choices": [{"message": {"content": reply_text}}]}
    return fake_post


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)


def test_gemini_plain_text(monkeypatch, keys):
    calls = []
    monkeypatch.setattr(llm, "_post", fake_gemini("hello", calls))
    assert ask_llm("Say hello", "gemini", "gemini-test") == "hello"
    assert "models/gemini-test:generateContent" in calls[0]["url"]
    assert calls[0]["headers"]["x-goog-api-key"] == FAKE_KEY
    assert "responseMimeType" not in calls[0]["body"]["generationConfig"]


def test_gemini_json_with_schema(monkeypatch, keys):
    calls = []
    monkeypatch.setattr(llm, "_post", fake_gemini('{"name": "Ada", "age": 36}', calls))
    assert ask_llm("Who?", "gemini", "gemini-test", schema=Person) == {"name": "Ada", "age": 36}
    assert calls[0]["body"]["generationConfig"]["responseMimeType"] == "application/json"


def test_groq_json_with_schema(monkeypatch, keys):
    calls = []
    monkeypatch.setattr(llm, "_post", fake_groq('{"name": "Ada", "age": 36}', calls))
    result = ask_llm("Who?", "groq", "llama-test", schema=Person)
    assert result == {"name": "Ada", "age": 36}
    assert calls[0]["body"]["model"] == "llama-test"
    assert calls[0]["body"]["response_format"] == {"type": "json_object"}
    assert calls[0]["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"


def test_json_in_code_fences_is_accepted():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_bad_json_raises(monkeypatch, keys):
    monkeypatch.setattr(llm, "_post", fake_gemini("Sure! Here is the data: name=Ada", []))
    with pytest.raises(LLMError, match="valid JSON"):
        ask_llm("Who?", "gemini", "gemini-test", schema=Person)


def test_schema_mismatch_raises(monkeypatch, keys):
    monkeypatch.setattr(llm, "_post", fake_gemini('{"name": "Ada", "age": "old"}', []))
    with pytest.raises(LLMError, match="schema"):
        ask_llm("Who?", "gemini", "gemini-test", schema=Person)


def test_missing_key_raises_without_leaking(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GROQ_API_KEY is not set"):
        ask_llm("hi", "groq", "llama-test")


def test_unknown_provider_raises():
    with pytest.raises(LLMError, match="unknown provider"):
        ask_llm("hi", "openai", "gpt-test")


def test_real_network_is_blocked_in_tests(keys):
    # conftest.py replaces _post for every test, so this can never reach the API
    with pytest.raises(LLMError, match="disabled in tests"):
        ask_llm("hi", "gemini", "gemini-test")


# ---------- retry on "busy" errors (requests.post is faked, no network) ----------

class FakeResponse:
    def __init__(self, status, payload=None):
        self.status_code = status
        self.text = "busy" if status != 200 else "ok"
        self._payload = payload or {}

    def json(self):
        return self._payload


def fake_requests(monkeypatch, statuses):
    calls = []
    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(url)
        return FakeResponse(statuses[len(calls) - 1], {"ok": True})
    monkeypatch.setattr(llm.requests, "post", fake_post)
    monkeypatch.setattr(llm, "RETRY_WAIT_SECONDS", 0)
    return calls


def test_retries_once_after_busy_error(monkeypatch):
    calls = fake_requests(monkeypatch, [503, 200])
    assert REAL_POST("https://x", {}, {}) == {"ok": True}
    assert len(calls) == 2


def test_gives_up_after_retry(monkeypatch):
    calls = fake_requests(monkeypatch, [429, 429])
    with pytest.raises(LLMError, match="HTTP 429"):
        REAL_POST("https://x", {}, {})
    assert len(calls) == 2


def test_no_retry_on_client_error(monkeypatch):
    calls = fake_requests(monkeypatch, [404])
    with pytest.raises(LLMError, match="HTTP 404"):
        REAL_POST("https://x", {}, {})
    assert len(calls) == 1
