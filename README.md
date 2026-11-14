# INTERVAI

A multi-agent AI mock interview platform. This README covers **Agent 1
(Resume Parser)**, **Agent 3 (Job Description Synthesizer + Match Scoring)**
and the **pipeline / API** that connects them to the HITL dashboard and Agent 4.

## Setup

```bash
conda activate intervai_env
pip install -r requirements.txt
cp .env.example .env        # then put your API keys in .env (never commit it)
```

Run every command below from the project root.

### LLM setup (`agents/llm.py`)

Every agent calls an LLM through one shared helper,
`ask_llm(prompt, provider, model, schema=None)`:

- **Providers:** `"gemini"` (key `GEMINI_API_KEY`) and `"groq"` (key `GROQ_API_KEY`),
  read from `.env` with python-dotenv. Keys are never printed or committed.
- **Per-agent model:** each agent sets its provider and model in one
  `LLM_CONFIG` dict at the top of its file, so switching models is a
  one-line change. Agent 1 uses `{"provider": "gemini", "model": "gemini-3.5-flash"}`.
- **Validated JSON:** pass a pydantic model as `schema` and the reply is
  parsed as JSON and checked against it. Any problem (missing key, network
  error, bad JSON, wrong fields) raises one error type, `LLMError`, so the
  agent can fall back cleanly.
- **Retries:** "busy" API errors (HTTP 429/5xx) are retried once after a
  short wait.

---

## Agent 1 — Resume Parser (`agents/resume_parser.py`)

Turns a resume file (`.pdf` or `.docx`) into a structured profile.
Agent 1 is an **LLM agent (Gemini) with a rule-based fallback**. Both
paths return exactly the same profile format.

### LLM path — `parse_resume_llm(path)` (default)

1. Reads the resume text (same readers as below).
2. Sends it to Gemini with a prompt that asks for a JSON profile. The prompt
   includes the skill list from `data/skills.txt`, so known skills come back
   with the same spelling Agent 3 matches on ("python" → "Python").
3. Checks the reply against a pydantic schema: name required, job dates
   must be `YYYY-MM` or `Present`, and so on.
4. **The LLM only reads; Python does the math.** The LLM returns each job's
   start and end month, and the code calculates months per job (inclusive) and
   total years, the same way as the rule-based parser. Language models are
   unreliable at date arithmetic.
5. Sets `"parsed_by": "llm"`.

**Fallback:** if the key is missing, the API is down or busy, the reply
isn't valid JSON, or it fails validation (e.g. a job that ends before it
starts), Agent 1 logs a warning and returns the rule-based result instead,
with `"parsed_by": "rules"`. The pipeline never breaks because of the LLM.

### Rule-based path — `parse_resume(path)` (fallback)

1. Reads the text (`pdfplumber` for PDF, `python-docx` for Word, including table cells).
2. Splits it into sections by headings in ALL CAPS or Title Case
   ("EXPERIENCE", "Work Experience", "Technical Skills:").
3. Extracts:
   - **Contact info** from the header: name (first line), email, phone, LinkedIn, GitHub.
   - **Skills** by matching the keyword list in `data/skills.txt` (whole-word
     match, so "Java" doesn't match "JavaScript"). URLs are removed first,
     so a `github.com/...` link doesn't count as the skill "GitHub".
   - **Education**: institution, degree, field, graduation date.
   - **Experience**: each job's company, title, dates and length in months,
     plus total years. Dates can be `Aug 2024`, `08/2024` or `2024`, and a
     range can end in `Present`. Months are counted inclusively
     (Jan–Dec 2023 = 12).

**Run it**
```bash
python agents/resume_parser.py                      # LLM, uses data/sample.pdf
python agents/resume_parser.py path/to/resume.docx  # LLM, your file
python agents/resume_parser.py --rules              # rule-based only, no API call
```
Prints the profile and saves it to `output/profile.json`.

**Compare the two parsers**
```bash
python scripts/compare_parsers.py      # 3 fake fixtures + data/sample.pdf (if present)
```
Prints only the fields where the LLM and the rules disagree. Results from
our run (identical across two runs):

| Resume | Differences |
|---|---|
| `resume_title_case.docx` | Rules found **Excel**, the LLM didn't. It's only mentioned in a job bullet ("records in Excel"), not in the skills list. |
| `resume_numeric_dates.docx` | identical |
| `resume_comma_skills.docx` | identical |
| `data/sample.pdf` | The LLM also found **Word** (from "Microsoft Excel, Word"). Rules can't, because "Word" isn't in `skills.txt`. |

Name, contact info, education, jobs, months and total years matched on
all four. The rule-based parser is already tuned for these formats, so the
LLM's advantage is on layouts the rules don't handle (see Known limitations).

**Output format**
```json
{
  "name": "Priya Nandakumar",
  "email": "priya.nandakumar@example.com",
  "phone": "(555) 201-3344",
  "linkedin": "linkedin.com/in/...",
  "github": null,
  "skills": ["Python", "SQL", "Tableau", "pandas"],
  "education": {"institution": "...", "degree": "Bachelor of Science in Statistics",
                "field": "Statistics", "graduation": "May 2022"},
  "experience": [{"company": "Brightline Analytics", "title": "Junior Data Analyst",
                  "dates": "Jan 2022 - Dec 2023", "months": 24}],
  "experience_years": 3.3,
  "parsed_by": "llm",
  "raw_text": "full resume text ..."
}
```
`parsed_by` is `"llm"` or `"rules"`, so you can always tell which path produced a profile.
`raw_text` is saved in the file (Agent 3 needs it) but left out of the
printout and out of the pipeline/API result.

---

## Agent 3 — Job Synthesizer + Match Scoring (`agents/job_synthesizer.py`)

Reads job postings, works out what each one requires, scores how well a
resume profile fits each job, and ranks them.

**Run it**
```bash
python agents/job_synthesizer.py                                   # sample.pdf + fake jobs
python agents/job_synthesizer.py path/to/resume.pdf path/to/jobs.json
```
Prints a ranking table and saves it to `output/ranked_jobs.json`.

**Job input format** — a JSON list of postings (all values are strings):
```json
[
  {
    "id": "job-001",
    "title": "Junior Java Developer",
    "company": "Maplewood Software",
    "location": "Monroe, LA",
    "url": "https://example.com/jobs/job-001",
    "description": "Full job description text ..."
  }
]
```
Only `description` is used to find requirements. The other fields are passed
through for display. `tests/fixtures/jobs.json` has 6 fake examples.

**Requirement extraction** (`extract_requirements(job)`) returns:
- `skills`: from the same `data/skills.txt` list Agent 1 uses
- `min_years`: the largest "N years" / "N+ years" in the text (0 if none)
- `degree`: the lowest degree mentioned (Associate < Bachelor < Master < PhD),
  since "Bachelor's or Master's" means a Bachelor's is enough, or `null`

**Output format** (`rank_jobs`) — best match first:
```json
{
  "id": "job-002",
  "title": "Software Engineering Intern",
  "company": "...", "location": "...", "url": "...",
  "score": 88.5,
  "matched_skills": ["Java", "C++", "Git", "GitHub", "Data Structures", "Algorithms"],
  "missing_skills": ["Linux"],
  "reason": "Matches 6/7 skills; missing Linux.",
  "breakdown": {"skill_points": 51.4, "text_similarity": 0.125, "text_points": 37.1,
                "required_years": 0, "candidate_years": 2.6, "experience_penalty": 0,
                "required_degree": "Bachelor", "candidate_degree": "Bachelor",
                "degree_penalty": 0}
}
```

### How scoring works

Each job gets a score from 0 to 100:

| Part | Points | How |
|---|---|---|
| Skill overlap | up to **60** | 60 × (matched skills ÷ skills the job lists) |
| Text similarity | up to **40** | TF-IDF cosine similarity, rescaled (see below) |
| Experience gap | **−10 per year** short, max −30 | if your total years < the job's minimum |
| Degree gap | **−5** | if the job asks for a higher degree than you have |

The final score is kept between 0 and 100.

**Why these weights?** Skills are the most direct signal (recruiters screen
on "do you have the listed skills"), so they count most. Text similarity
catches overlap the skill list misses, like job titles, field of study, or words
like "backend" and "data", but it's noisier, so it counts less. The
experience penalty pushes senior roles down the list without removing them.
The degree penalty is small because students often apply while still
finishing a degree.

**TF-IDF similarity and the "relative" tradeoff.** TF-IDF is fitted on the
resume's full text **plus all the job descriptions together**, so words that
appear in every posting ("team", "experience") count less than specific
ones ("PyTorch", "backend"). Raw cosine similarity between a resume and a
job posting is small (about 0.03–0.3), so it's rescaled: the most similar
job gets the full 40 points and every other job gets
`40 × its similarity ÷ the best similarity`.

- *Benefit:* scores spread across the full 0–100 range and are easy to read.
- *Tradeoff:* text points are **relative to the other jobs in the list**.
  Adding or removing a posting can change a job's score even though the
  job itself didn't change. Scores should be compared within one ranking,
  not across different job lists. Calling `score_match(profile, job)` on a
  single job gives it the full 40 text points if it shares any words with
  the resume, so use `rank_jobs` for real rankings.

---

## Pipeline + API (connecting to the dashboard and Agent 4)

**Python entry point** (`agents/pipeline.py`):
```python
from pipeline import run_pipeline
result = run_pipeline("data/sample.pdf", jobs)   # jobs = list of job dicts
result["profile"]       # Agent 1 output, without raw_text
result["ranked_jobs"]   # Agent 3 output, best match first

run_pipeline("data/sample.pdf", jobs, use_llm=False)   # rules only, no API call
```
It checks the job list first and raises `ValueError` if a job is missing fields.
By default Agent 1 uses the LLM (with fallback), and the API does the same.

**Web API** (`api.py`, FastAPI):
```bash
uvicorn api:app --reload
```
`POST /rank` takes a multipart form with `resume` (a `.pdf`/`.docx` file) and
`jobs` (a JSON list as text), and returns `{"profile": ..., "ranked_jobs": [...]}`.
```bash
curl -X POST http://127.0.0.1:8000/rank \
     -F "resume=@data/sample.pdf" \
     -F "jobs=<tests/fixtures/jobs.json"
```
Interactive docs: http://127.0.0.1:8000/docs. Bad file types, invalid JSON
or incomplete jobs return HTTP 400. The uploaded resume goes to a temp file
that's deleted as soon as the request finishes.

---

## Tests

```bash
python -m pytest -v
```

| File | Covers |
|---|---|
| `tests/test_llm.py` | `ask_llm`: Gemini/Groq requests, JSON + schema validation, missing keys, retries |
| `tests/test_parser.py` | Agent 1 rule-based parser on 3 fake resumes with different heading, date and skill styles |
| `tests/test_parser_llm.py` | Agent 1 LLM parser: same format as rules, cleanup, every fallback case, pipeline default |
| `tests/test_job_synthesizer.py` | Requirement extraction, scoring, penalties, ranking |
| `tests/test_pipeline.py` | `run_pipeline` and the `/rank` endpoint (FastAPI `TestClient`) |

**Tests never call a real LLM.** `tests/conftest.py` blocks the network
function in `agents/llm.py` for every test, and tests that need an LLM
reply install a fake one. The suite runs offline and without API keys.

All test data is fake: `tests/fixtures/` holds invented resumes (rebuild
them with `python tests/fixtures/make_fixtures.py`) and invented job postings.
The one test that uses `data/sample.pdf` is skipped when that file is missing.
Real resumes (`data/*.pdf`, `data/*.docx`) and `output/` are git-ignored.

---

## Known limitations

**Agent 1, LLM path**
- **Resume text is sent to Google (Gemini).** Real resumes are personal
  data, so use `--rules` / `use_llm=False` when that isn't acceptable.
- **Output can vary between runs**, even at temperature 0. The schema check
  catches broken output, but not a plausible wrong answer.
- **It can skip skills only mentioned in passing** (e.g. "Excel" in a job
  bullet), and it can add skills that aren't in `skills.txt`, which Agent 3
  will never match against a job.
- **Prompt injection:** the prompt tells the model to treat the resume as
  data, but a resume containing hidden instructions could still affect it.
- Each resume costs one API call and takes a few seconds.

**Agent 1, rule-based fallback**
- **Skills are keyword-only.** Only skills listed in `data/skills.txt` are
  found, with no synonyms ("JS" ≠ "JavaScript") and no skills implied
  from context.
- **Section headings must contain a known keyword** (Experience, Skills,
  Education, …). A heading like "Employment History" isn't recognized.
- **Experience assumes a layout**: the line above the dates is read as
  "Company | Title".
- **Contact and education are simple**: the name is the first header line,
  and only the first degree/institution is read.

**Agent 1, both paths**
- **No OCR.** Scanned or image-only PDFs give no text.
- Overlapping jobs are added together, and year-only dates are approximated
  (start = January, end = December).

**Agent 3**
- **Job requirement parsing is regex-based**: `min_years` takes the largest
  "N years" in the text, even if it isn't about experience (e.g. "founded
  10 years ago"). "MS"/"M.S." only counts as a Master's in a degree context
  ("MS in …", "M.S. degree", "Degree: MS", "BS/MS"), so "MS Excel" is ignored,
  but an unusual phrasing like "MS required" is missed.
- **TF-IDF is bag-of-words.** It matches shared words, not meaning
  ("developer" vs. "engineer" don't match), and its scores are relative
  to the job list (see above).
- **The API has no authentication** and is meant for local use and demos.
