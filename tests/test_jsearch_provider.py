"""Tests for Agent 2's JSearch provider (jsearch_provider.py).

No real request is ever made: requests.get is replaced with a fake that
returns canned JSON. All data here is fake.

Run from project root:  python -m pytest -v
"""
import os
import sys

import pytest
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "agents"))

import jsearch_provider as jp  # noqa: E402
from job_search import JOB_FIELDS, search_jobs  # noqa: E402
from pipeline import validate_jobs  # noqa: E402

KEY = "SECRET-TEST-KEY-123"


def raw_job(n, **overrides):
    job = {"job_id": f"long-id-{n}", "job_uid": f"uid-{n}", "job_title": f"Python Developer {n}",
           "employer_name": "Acme", "job_apply_link": f"https://x.test/{n}",
           "job_description": "Build things in Python.", "job_location": "Houma, LA",
           "job_city": "Houma", "job_state": "Louisiana", "job_is_remote": False}
    job.update(overrides)
    return job


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code, self.body = status, body

    def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


def v2(*jobs):
    """A /search-v2 style body: data is a dict holding the jobs list."""
    return {"status": "OK", "data": {"jobs": list(jobs), "cursor": "abc"}}


def fake_get(monkeypatch, response):
    """Replaces requests.get. Returns a dict that records the last call."""
    seen = {}

    def get(url, headers=None, params=None, timeout=None):
        seen.update(url=url, headers=headers, params=params)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(jp.requests, "get", get)
    return seen


@pytest.fixture
def provider():
    return jp.JSearchProvider(api_key=KEY)


# ---------- building the request ----------

def test_query_text_combines_title_keywords_and_location(provider):
    q = {"title": "Data Analyst", "keywords": ["Python", "SQL"], "location": "Louisiana"}
    assert provider.build_query_text(q) == "Data Analyst Python SQL in Louisiana"

def test_query_text_keeps_only_the_first_two_keywords(provider):
    q = {"title": "Software Developer", "keywords": ["Java", "Python", "C", "C++"]}
    assert provider.build_query_text(q) == "Software Developer Java Python"

def test_query_text_without_location(provider):
    assert provider.build_query_text({"keywords": ["Python"]}) == "Python"


def test_empty_query_still_gives_a_search(provider):
    assert provider.build_query_text({}) == "jobs"


def test_request_uses_search_v2_host_key_and_one_page(monkeypatch, provider):
    seen = fake_get(monkeypatch, FakeResponse(200, v2(raw_job(1))))
    provider.search({"keywords": ["Python"]}, 20)
    assert seen["url"].endswith("/search-v2")
    assert seen["headers"]["X-RapidAPI-Key"] == KEY
    assert seen["headers"]["X-RapidAPI-Host"] == "jsearch.p.rapidapi.com"
    assert seen["params"]["query"] == "Python"
    assert seen["params"]["num_pages"] == 1      # free plan: keep it to one request


# ---------- parsing results ----------

def test_parses_v2_jobs_into_raw_dicts(monkeypatch, provider):
    fake_get(monkeypatch, FakeResponse(200, v2(raw_job(1))))
    job = provider.search({}, 10)[0]
    assert job == {"id": "uid-1", "title": "Python Developer 1", "company": "Acme",
                   "location": "Houma, LA", "url": "https://x.test/1",
                   "description": "Build things in Python."}


def test_plain_list_data_also_works(monkeypatch, provider):
    fake_get(monkeypatch, FakeResponse(200, {"status": "OK", "data": [raw_job(1), raw_job(2)]}))
    assert len(provider.search({}, 10)) == 2


def test_empty_results_give_empty_list(monkeypatch, provider):
    fake_get(monkeypatch, FakeResponse(200, v2()))
    assert provider.search({}, 10) == []


def test_missing_data_gives_empty_list(monkeypatch, provider):
    fake_get(monkeypatch, FakeResponse(200, {"status": "OK"}))
    assert provider.search({}, 10) == []


def test_location_falls_back_to_city_and_state():
    job = jp.parse_job(raw_job(1, job_location=None))
    assert job["location"] == "Houma, Louisiana"


def test_remote_job_without_location_is_labelled_remote():
    job = jp.parse_job(raw_job(1, job_location=None, job_city=None,
                               job_state=None, job_is_remote=True))
    assert job["location"] == "Remote"


def test_id_falls_back_to_job_id_when_uid_missing():
    assert jp.parse_job(raw_job(1, job_uid=None))["id"] == "long-id-1"


# ---------- working with search_jobs ----------

def test_results_pass_search_jobs_and_pipeline_validation(monkeypatch, provider):
    fake_get(monkeypatch, FakeResponse(200, v2(raw_job(1), raw_job(2))))
    jobs = search_jobs({}, provider)
    validate_jobs(jobs)  # raises if the shape is wrong
    assert tuple(jobs[0].keys()) == JOB_FIELDS
    assert jobs[0]["id"] == "jsearch-uid-1"


def test_incomplete_and_duplicate_jobs_are_dropped(monkeypatch, provider):
    fake_get(monkeypatch, FakeResponse(200, v2(
        raw_job(1), raw_job(1), raw_job(2, job_description=""), raw_job(3, job_title=""))))
    assert len(search_jobs({}, provider)) == 1


# ---------- errors ----------

def test_missing_key_raises_without_making_a_request(monkeypatch):
    monkeypatch.delenv("RAPIDAPI_KEY", raising=False)
    seen = fake_get(monkeypatch, FakeResponse(200, v2(raw_job(1))))
    with pytest.raises(jp.ProviderError, match="RAPIDAPI_KEY"):
        jp.JSearchProvider(api_key="").search({}, 10)
    assert seen == {}


@pytest.mark.parametrize("status", [401, 403])
def test_rejected_key_gives_clear_error(monkeypatch, provider, status):
    fake_get(monkeypatch, FakeResponse(status, {}))
    with pytest.raises(jp.ProviderError, match="rejected the key"):
        provider.search({}, 10)


def test_quota_or_rate_limit_error(monkeypatch, provider):
    fake_get(monkeypatch, FakeResponse(429, {}))
    with pytest.raises(jp.ProviderError, match="429"):
        provider.search({}, 10)


def test_other_http_errors(monkeypatch, provider):
    fake_get(monkeypatch, FakeResponse(500, {}))
    with pytest.raises(jp.ProviderError, match="HTTP 500"):
        provider.search({}, 10)


def test_network_error_becomes_provider_error(monkeypatch, provider):
    fake_get(monkeypatch, requests.ConnectionError("boom"))
    with pytest.raises(jp.ProviderError, match="network error: ConnectionError"):
        provider.search({}, 10)


def test_invalid_json_becomes_provider_error(monkeypatch, provider):
    fake_get(monkeypatch, FakeResponse(200, ValueError("bad json")))
    with pytest.raises(jp.ProviderError, match="valid JSON"):
        provider.search({}, 10)


@pytest.mark.parametrize("response", [
    FakeResponse(401, {}), FakeResponse(429, {}), FakeResponse(500, {}),
    FakeResponse(200, ValueError("bad")), requests.ConnectionError("boom"),
])
def test_api_key_never_appears_in_error_messages(monkeypatch, provider, response):
    fake_get(monkeypatch, response)
    with pytest.raises(jp.ProviderError) as error:
        provider.search({}, 10)
    assert KEY not in str(error.value)