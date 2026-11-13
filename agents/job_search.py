"""Agent 2: Web Search & Opportunity Agent.

Fetches job postings from a provider and returns them in the exact shape
Agent 3 / run_pipeline() expects:
    {id, title, company, location, url, description}   (all strings)

    from job_search import search_jobs
    jobs = search_jobs({"keywords": ["Python", "SQL"]}, provider, limit=20)
"""
import hashlib

JOB_FIELDS = ("id", "title", "company", "location", "url", "description")


class JobProvider:
    """Base class for job sources. Subclasses return a list of raw dicts."""
    name = "base"

    def search(self, query, limit):
        raise NotImplementedError


def normalize_job(raw, source):
    """Turns one raw posting into the standard job dict, or None if it
    has no title or description (Agent 3 can't score it without them)."""
    job = {f: str(raw.get(f) or "").strip() for f in JOB_FIELDS}
    if not job["title"] or not job["description"]:
        return None
    if job["id"]:
        job["id"] = f"{source}-{job['id']}"
    else:
        basis = job["url"] or (job["title"] + job["company"])
        job["id"] = f"{source}-{hashlib.sha1(basis.encode()).hexdigest()[:10]}"
    return job


def search_jobs(query, provider, limit=20):
    """Runs the provider, cleans the results, removes duplicates (same url,
    or same title + company when there is no url) and caps at `limit`."""
    jobs, seen = [], set()
    for raw in provider.search(query, limit):
        job = normalize_job(raw, provider.name)
        if job is None:
            continue
        key = job["url"] or (job["title"].lower(), job["company"].lower())
        if key in seen:
            continue
        seen.add(key)
        jobs.append(job)
        if len(jobs) >= limit:
            break
    return jobs