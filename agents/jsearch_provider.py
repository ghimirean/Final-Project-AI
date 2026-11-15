"""JSearch (RapidAPI) job provider for Agent 2.

Plugs into search_jobs() as a JobProvider: it turns Agent 2's query dict into
a JSearch search request and returns raw job dicts (id, title, company,
location, url, description) for normalize_job() to clean.

The key comes from .env (see .env.example): RAPIDAPI_KEY. It is sent in a
request header and never appears in error messages.
"""
import os

import requests
from dotenv import load_dotenv

from job_search import JobProvider

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

URL = "https://jsearch.p.rapidapi.com/search-v2"
HOST = "jsearch.p.rapidapi.com"
TIMEOUT_SECONDS = 30


class ProviderError(Exception):
    """Raised for any job-source problem: missing key, network or HTTP
    error, or a reply that isn't the expected JSON."""


class JSearchProvider(JobProvider):
    name = "jsearch"

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("RAPIDAPI_KEY")

    def build_query_text(self, query):
        """Agent 2 query dict -> one free-text JSearch query, e.g.
        'Data Analyst Python SQL in Remote'. JSearch has no experience-level
        field here, so that part of the query is not sent."""
        words = ([query["title"]] if query.get("title") else []) + list(query.get("keywords") or [])
        text = " ".join(words) or "jobs"
        if query.get("location"):
            text += f" in {query['location']}"
        return text

    def search(self, query, limit):
        # One request per search (one page), because the free plan only
        # allows 200 requests a month. `limit` is applied by search_jobs().
        if not self.api_key:
            raise ProviderError("RAPIDAPI_KEY is not set (add to .env)")
        headers = {"X-RapidAPI-Key": self.api_key, "X-RapidAPI-Host": HOST}
        params = {"query": self.build_query_text(query), "num_pages": 1, "country": "us"}
        try:
            response = requests.get(URL, headers=headers, params=params,
                                    timeout=TIMEOUT_SECONDS)
        except requests.RequestException as e:
            raise ProviderError(f"network error: {type(e).__name__}") from None
        if response.status_code in (401, 403):
            raise ProviderError("JSearch rejected the key (check RAPIDAPI_KEY and "
                                f"that you subscribed to the plan): HTTP {response.status_code}")
        if response.status_code == 429:
            raise ProviderError("JSearch rate limit or monthly quota reached (HTTP 429)")
        if response.status_code != 200:
            raise ProviderError(f"JSearch returned HTTP {response.status_code}")
        try:
            data = response.json().get("data") or []
        except (ValueError, AttributeError):
            raise ProviderError("JSearch did not return valid JSON") from None
        # /search-v2 returns {"jobs": [...], "cursor": "..."}; a plain list also works
        jobs = data.get("jobs") if isinstance(data, dict) else data
        return [parse_job(item) for item in (jobs or []) if isinstance(item, dict)]


def parse_job(item):
    """One JSearch result -> the raw dict shape normalize_job() expects.
    job_uid is used as the id because job_id is a very long string."""
    location = item.get("job_location") or ", ".join(
        p for p in (item.get("job_city"), item.get("job_state")) if p)
    if not location and item.get("job_is_remote"):
        location = "Remote"
    return {
        "id": item.get("job_uid") or item.get("job_id"),
        "title": item.get("job_title"),
        "company": item.get("employer_name"),
        "location": location,
        "url": item.get("job_apply_link"),
        "description": item.get("job_description"),
    }