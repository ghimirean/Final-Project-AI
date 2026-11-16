"""Small web API around the Agent 1 + Agent 3 pipeline.

Start it from the project root:
    uvicorn api:app --reload

Then try it (or open http://127.0.0.1:8000/docs for a clickable form):
    curl -X POST http://127.0.0.1:8000/rank \
         -F "resume=@data/sample.pdf" \
         -F "jobs=<tests/fixtures/jobs.json"
"""
import json
import os
import sys
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents"))
from pipeline import run_pipeline  # noqa: E402
from job_search_agent import run_job_agent  # noqa: E402
from job_synthesizer import SKILLS_PATH  # noqa: E402
from jsearch_provider import JSearchProvider, ProviderError  # noqa: E402
from resume_parser import parse_resume_llm  # noqa: E402

app = FastAPI(title="INTERVAI - Resume parser + job ranking")

ALLOWED_TYPES = {".pdf", ".docx"}
MAX_SEARCH_LIMIT = 50


@app.post("/rank")
async def rank(resume: UploadFile = File(...), jobs: str = Form(...)):
    """resume: a .pdf or .docx file. jobs: a JSON list of job postings
    ({id, title, company, location, url, description}).
    Returns {"profile": ..., "ranked_jobs": [...]}.
    """
    ext = os.path.splitext(resume.filename or "")[1].lower()
    if ext not in ALLOWED_TYPES:
        raise HTTPException(400, f"resume must be .pdf or .docx, got '{ext or 'no extension'}'")
    try:
        job_list = json.loads(jobs)
    except json.JSONDecodeError:
        raise HTTPException(400, "jobs must be valid JSON")

    # The parser reads from a file path, so save the upload to a temp file
    # and delete it right after (resumes are personal data; don't keep them)
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await resume.read())
        tmp_path = tmp.name
    try:
        return run_pipeline(tmp_path, job_list)
    except ValueError as e:                 # bad jobs list
        raise HTTPException(400, str(e))
    finally:
        os.remove(tmp_path)

@app.post("/search")
async def search(resume: UploadFile = File(...), location: str = Form(""),
                 limit: int = Form(20)):
    """resume: a .pdf or .docx file. location: optional city/state text.
    limit: most jobs to return (1 to 50, default 20).
    Agent 1 parses the resume, then Agent 2 (an LLM agent) searches for jobs.
    Returns {"profile": ..., "jobs": [...], "search_mode": "llm" or "rules",
    "trace": [...]}. "jobs" has the format /rank expects, but it can be empty
    (and /rank rejects an empty list), so check it before ranking.
    """
    ext = os.path.splitext(resume.filename or "")[1].lower()
    if ext not in ALLOWED_TYPES:
        raise HTTPException(400, f"resume must be .pdf or .docx, got '{ext or 'no extension'}'")
    if not 1 <= limit <= MAX_SEARCH_LIMIT:
        raise HTTPException(400, f"limit must be between 1 and {MAX_SEARCH_LIMIT}")

    # Same temp-file handling as /rank: the parser needs a path, and the
    # resume (personal data) is deleted as soon as it has been parsed.
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await resume.read())
        tmp_path = tmp.name
    try:
        profile = parse_resume_llm(tmp_path, skills_path=SKILLS_PATH)
    finally:
        os.remove(tmp_path)

    try:
        result = run_job_agent(profile, JSearchProvider(), location=location, limit=limit)
    except ProviderError as e:              # job source failed on the rule-based fallback
        raise HTTPException(502, str(e))
    if not result["jobs"] and any("search_error" in step for step in result["trace"]):
        # the LLM agent's searches all failed (bad key, quota used up, site down)
        raise HTTPException(502, "the job source returned an error "
                                 "(check RAPIDAPI_KEY and your monthly request quota)")

    public_profile = {k: v for k, v in profile.items() if k != "raw_text"}
    return {"profile": public_profile, "jobs": result["jobs"],
            "search_mode": result["mode"], "trace": result["trace"]}
