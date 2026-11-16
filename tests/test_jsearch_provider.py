"""Tests for Agent 2's JSearch provider (jsearch_provider.py).

No real request is ever made: requests.get is replaced with a fake that
returns canned JSON. All data here is fake.

Run from project root:  python -m pytest -v
"""
import json
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
    # cache_path=None: these tests never read or write a real cache file
    return jp.JSearchProvider(api_key=KEY, cache_path=None)


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
        jp.JSearchProvider(api_key="", cache_path=None).search({}, 10)
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


# ---------- on-disk cache (tmp_path only, never output/) ----------

QUERY = {"title": "Data Analyst", "keywords": ["Python"], "location": "Louisiana"}
QUERY_TEXT = "Data Analyst Python in Louisiana"


def counting_get(monkeypatch, *responses):
    """Replaces requests.get with one that returns the given responses in
    order and counts the calls."""
    calls = []
    responses = list(responses)

    def get(url, headers=None, params=None, timeout=None):
        calls.append(params["query"])
        return responses.pop(0)

    monkeypatch.setattr(jp.requests, "get", get)
    return calls


@pytest.fixture
def cache_file(tmp_path):
    return str(tmp_path / "jsearch_cache.json")


@pytest.fixture
def cached_provider(cache_file):
    return jp.JSearchProvider(api_key=KEY, cache_path=cache_file)


def test_default_cache_path_is_in_output():
    assert jp.DEFAULT_CACHE_PATH.endswith(os.path.join("output", "jsearch_cache.json"))
    assert jp.JSearchProvider(api_key=KEY).cache_path == jp.DEFAULT_CACHE_PATH


def test_cache_hit_skips_the_http_call(monkeypatch, cached_provider, cache_file):
    calls = counting_get(monkeypatch, FakeResponse(200, v2(raw_job(1))))
    first = cached_provider.search(QUERY, 10)
    second = cached_provider.search(QUERY, 10)
    assert calls == [QUERY_TEXT]                      # only one real request
    assert second == first and first[0]["id"] == "uid-1"
    with open(cache_file, encoding="utf-8") as f:
        assert list(json.load(f)) == [QUERY_TEXT]     # keyed by the final query text


def test_cache_is_shared_between_provider_instances(monkeypatch, cache_file):
    calls = counting_get(monkeypatch, FakeResponse(200, v2(raw_job(1))))
    jp.JSearchProvider(api_key=KEY, cache_path=cache_file).search(QUERY, 10)
    jp.JSearchProvider(api_key=KEY, cache_path=cache_file).search(QUERY, 10)
    assert len(calls) == 1


def test_different_query_is_not_a_cache_hit(monkeypatch, cached_provider):
    calls = counting_get(monkeypatch, FakeResponse(200, v2(raw_job(1))),
                         FakeResponse(200, v2(raw_job(2))))
    cached_provider.search(QUERY, 10)
    cached_provider.search({**QUERY, "location": "Texas"}, 10)
    assert len(calls) == 2


def test_expired_entry_is_refreshed(monkeypatch, cached_provider):
    calls = counting_get(monkeypatch, FakeResponse(200, v2(raw_job(1))),
                         FakeResponse(200, v2(raw_job(2))))
    now = [1_000_000.0]
    monkeypatch.setattr(jp.time, "time", lambda: now[0])
    cached_provider.search(QUERY, 10)
    now[0] += jp.CACHE_TTL_SECONDS - 1                # still fresh
    assert cached_provider.search(QUERY, 10)[0]["id"] == "uid-1"
    now[0] += 2                                       # now older than 24 hours
    assert cached_provider.search(QUERY, 10)[0]["id"] == "uid-2"
    assert len(calls) == 2
    now[0] += 1                                       # the refreshed entry is used
    assert cached_provider.search(QUERY, 10)[0]["id"] == "uid-2"
    assert len(calls) == 2


@pytest.mark.parametrize("response", [
    FakeResponse(429, {}), FakeResponse(500, {}), FakeResponse(200, ValueError("bad")),
])
def test_errors_are_not_cached(monkeypatch, cached_provider, cache_file, response):
    calls = counting_get(monkeypatch, response, FakeResponse(200, v2(raw_job(1))))
    with pytest.raises(jp.ProviderError):
        cached_provider.search(QUERY, 10)
    assert not os.path.exists(cache_file)
    assert len(cached_provider.search(QUERY, 10)) == 1   # next call asks JSearch again
    assert len(calls) == 2


def test_empty_results_are_not_cached(monkeypatch, cached_provider, cache_file):
    calls = counting_get(monkeypatch, FakeResponse(200, v2()),
                         FakeResponse(200, v2(raw_job(1))))
    assert cached_provider.search(QUERY, 10) == []
    assert not os.path.exists(cache_file)
    assert len(cached_provider.search(QUERY, 10)) == 1
    assert len(calls) == 2


@pytest.mark.parametrize("content", ["{not json", "[1, 2, 3]",
                                     '{"Data Analyst Python in Louisiana": "oops"}'])
def test_corrupt_cache_file_is_ignored(monkeypatch, cached_provider, cache_file, content):
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(content)
    calls = counting_get(monkeypatch, FakeResponse(200, v2(raw_job(1))))
    assert cached_provider.search(QUERY, 10)[0]["id"] == "uid-1"
    assert len(calls) == 1
    with open(cache_file, encoding="utf-8") as f:     # rewritten as a valid cache
        assert QUERY_TEXT in json.load(f)


def test_no_cache_when_cache_path_is_none(monkeypatch, provider):
    calls = counting_get(monkeypatch, FakeResponse(200, v2(raw_job(1))),
                         FakeResponse(200, v2(raw_job(1))))
    provider.search(QUERY, 10)
    provider.search(QUERY, 10)
    assert len(calls) == 2