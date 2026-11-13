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

app = FastAPI(title="INTERVAI - Resume parser + job ranking")

ALLOWED_TYPES = {".pdf", ".docx"}


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
