"""Single entry point for Agent 1 + Agent 3.

The HITL dashboard and Agent 4 should call run_pipeline() instead of
calling the agents one by one.

    from pipeline import run_pipeline
    result = run_pipeline("data/sample.pdf", jobs)
    result["profile"]       # Agent 1 output (without raw_text)
    result["ranked_jobs"]   # Agent 3 output, best match first
"""
from resume_parser import parse_resume, parse_resume_llm, parse_resume_ner
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


PARSERS = {"ner": parse_resume_ner, "llm": parse_resume_llm, "rules": parse_resume}


def run_pipeline(resume_path, jobs, use_llm=True, parser="ner", llm_key_points=False):
    """Parses a resume (.pdf or .docx) and ranks the given jobs for it.
    By default everything runs locally: Agent 1 = NER model, Agent 3 =
    embeddings (scores + KeyBERT key points). No API calls.
    parser: how Agent 1 reads the resume: "ner" (default, local NER model),
        "llm" (Gemini) or "rules". NER and LLM fall back to rules on failure.
    llm_key_points=True (optional): Agent 3 asks Groq for the top jobs' key points.
    use_llm=False: forbids all API calls (an "llm" parser becomes "rules",
        and llm_key_points is ignored).
    Returns {"profile": {...}, "ranked_jobs": [...]}.
    """
    validate_jobs(jobs)
    if parser not in PARSERS:
        raise ValueError(f"parser must be one of {sorted(PARSERS)}")
    if parser == "llm" and not use_llm:
        parser = "rules"
    profile = PARSERS[parser](resume_path, skills_path=SKILLS_PATH)
    # uses raw_text for similarity
    ranked = rank_jobs(profile, jobs, use_llm=use_llm and llm_key_points)
    # raw_text is only needed for scoring; leave it out of the result
    public_profile = {k: v for k, v in profile.items() if k != "raw_text"}
    return {"profile": public_profile, "ranked_jobs": ranked}
