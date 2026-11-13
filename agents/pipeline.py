"""Single entry point for Agent 1 + Agent 3.

The HITL dashboard and Agent 4 should call run_pipeline() instead of
calling the agents one by one.

    from pipeline import run_pipeline
    result = run_pipeline("data/sample.pdf", jobs)
    result["profile"]       # Agent 1 output (without raw_text)
    result["ranked_jobs"]   # Agent 3 output, best match first
"""
from resume_parser import parse_resume
from job_synthesizer import rank_jobs, SKILLS_PATH

JOB_FIELDS = {"id", "title", "company", "location", "url", "description"}


def validate_jobs(jobs):
    # Fails early with a clear message instead of a confusing KeyError later
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs must be a non-empty list of job dicts")
    for i, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError(f"job #{i} is not a dict")
        missing = JOB_FIELDS - set(job)
        if missing:
            raise ValueError(f"job #{i} is missing fields: {sorted(missing)}")


def run_pipeline(resume_path, jobs):
    """Parses a resume (.pdf or .docx) and ranks the given jobs for it.
    Returns {"profile": {...}, "ranked_jobs": [...]}.
    """
    validate_jobs(jobs)
    profile = parse_resume(resume_path, skills_path=SKILLS_PATH)
    ranked = rank_jobs(profile, jobs)      # uses raw_text for TF-IDF
    # raw_text is only needed for scoring; leave it out of the result
    public_profile = {k: v for k, v in profile.items() if k != "raw_text"}
    return {"profile": public_profile, "ranked_jobs": ranked}
