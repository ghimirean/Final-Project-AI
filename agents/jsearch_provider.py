"""JSearch (RapidAPI) job provider for Agent 2.

Plugs into search_jobs() as a JobProvider: it turns Agent 2's query dict into
a JSearch search request and returns raw job dicts (id, title, company,
location, url, description) for normalize_job() to clean.

The key comes from .env (see .env.example): RAPIDAPI_KEY. It is sent in a
request header and never appears in error messages.

Results are cached on disk (output/jsearch_cache.json, git-ignored) for 24
hours, keyed by the final query text, so repeating a search doesn't use up
the free plan's 200 requests a month. Only successful, non-empty results are
cached: an empty result or an error is never stored, so the next run asks
JSearch again. Delete the file to get fresh results; pass cache_path=None to
turn the cache off.

After every search, provider.last_request says what happened, for the run
trace: {"query_text", "http_status" (None if cached or no reply),
"raw_count" (jobs in JSearch's reply), "from_cache"}.
"""
import json
import logging
import os
import time

import requests
from dotenv import load_dotenv

from job_search import JobProvider

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

URL = "https://jsearch.p.rapidapi.com/search-v2"
HOST = "jsearch.p.rapidapi.com"
TIMEOUT_SECONDS = 30
MAX_QUERY_KEYWORDS = 2    # long free-text queries often match nothing
DEFAULT_CACHE_PATH = os.path.join(ROOT, "output", "jsearch_cache.json")
CACHE_TTL_SECONDS = 24 * 60 * 60

log = logging.getLogger(__name__)


class ProviderError(Exception):
    """Raised for any job-source problem: missing key, network or HTTP
    error, or a reply that isn't the expected JSON. `status` is the HTTP
    status code, or None if there was no HTTP reply."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class JSearchProvider(JobProvider):
    name = "jsearch"

    def __init__(self, api_key=None, cache_path=DEFAULT_CACHE_PATH):
        self.api_key = api_key or os.environ.get("RAPIDAPI_KEY")
        self.cache_path = cache_path      # None = no caching
        self.last_request = None          # what the last search did (see top of file)

    def build_query_text(self, query):
        """Agent 2 query dict -> one free-text JSearch query, e.g.
        'Data Analyst Python SQL in Remote'. Only the first MAX_QUERY_KEYWORDS
        keywords are used, because long queries return no results. JSearch has
        no experience-level field here, so that part of the query is not sent."""
        keywords = list(query.get("keywords") or [])[:MAX_QUERY_KEYWORDS]
        words = ([query["title"]] if query.get("title") else []) + keywords
        text = " ".join(words) or "jobs"
        if query.get("location"):
            text += f" in {query['location']}"
        return text

    def search(self, query, limit):
        # One request per search (one page), because the free plan only
        # allows 200 requests a month. `limit` is applied by search_jobs().
        if not self.api_key:
            raise ProviderError("RAPIDAPI_KEY is not set (add to .env)")
        text = self.build_query_text(query)
        self.last_request = {"query_text": text, "http_status": None,
                             "raw_count": None, "from_cache": False}
        cached = self._cache_get(text)
        if cached is not None:
            self.last_request.update(raw_count=len(cached), from_cache=True)
            return cached
        jobs = self._fetch(text)
        if jobs:                          # never cache empty results (or errors)
            self._cache_put(text, jobs)
        return jobs

    def _fetch(self, text):
        headers = {"X-RapidAPI-Key": self.api_key, "X-RapidAPI-Host": HOST}
        params = {"query": text, "num_pages": 1, "country": "us"}
        try:
            response = requests.get(URL, headers=headers, params=params,
                                    timeout=TIMEOUT_SECONDS)
        except requests.RequestException as e:
            raise ProviderError(f"network error: {type(e).__name__}") from None
        status = response.status_code
        self.last_request["http_status"] = status
        if status in (401, 403):
            raise ProviderError("JSearch rejected the key (check RAPIDAPI_KEY and "
                                f"that you subscribed to the plan): HTTP {status}", status)
        if status == 429:
            raise ProviderError("JSearch rate limit or monthly quota reached (HTTP 429)", status)
        if status != 200:
            raise ProviderError(f"JSearch returned HTTP {status}", status)
        try:
            data = response.json().get("data") or []
        except (ValueError, AttributeError):
            raise ProviderError("JSearch did not return valid JSON (HTTP 200)", status) from None
        # /search-v2 returns {"data": {"jobs": [...], "cursor": ...}}; a plain list also works
        jobs = data.get("jobs") if isinstance(data, dict) else data
        jobs = [parse_job(item) for item in (jobs or []) if isinstance(item, dict)]
        self.last_request["raw_count"] = len(jobs)
        return jobs

    # ---------- on-disk cache ----------

    def _cache_load(self):
        """The whole cache as {query text: {"time", "jobs"}}. A missing or
        corrupt file counts as an empty cache: the cache must never break a search."""
        try:
            with open(self.cache_path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as e:
            log.warning("ignoring unreadable JSearch cache %s: %s", self.cache_path,
                        type(e).__name__)
            return {}
        return data if isinstance(data, dict) else {}

    def _cache_get(self, text):
        if self.cache_path is None:
            return None
        entry = self._cache_load().get(text)
        if not isinstance(entry, dict):
            return None
        saved, jobs = entry.get("time"), entry.get("jobs")
        if not isinstance(saved, (int, float)) or not isinstance(jobs, list) or not jobs:
            return None
        if time.time() - saved >= CACHE_TTL_SECONDS:
            return None                   # expired: search again
        return jobs

    def _cache_put(self, text, jobs):
        if self.cache_path is None:
            return
        now = time.time()
        cache = {k: v for k, v in self._cache_load().items()     # drop expired entries
                 if isinstance(v, dict) and isinstance(v.get("time"), (int, float))
                 and now - v["time"] < CACHE_TTL_SECONDS}
        cache[text] = {"time": now, "jobs": jobs}
        try:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            tmp = self.cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cache, f)
            os.replace(tmp, self.cache_path)   # no half-written cache file
        except OSError as e:
            log.warning("could not write JSearch cache %s: %s", self.cache_path,
                        type(e).__name__)


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