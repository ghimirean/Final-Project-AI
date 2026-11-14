# INTERVAI

A multi-agent AI mock interview platform. This README covers **Agent 1
(Resume Parser)**, **Agent 3 (Job Description Synthesizer + Match Scoring)**
and the **pipeline / API** that connects them to the HITL dashboard and Agent 4.

| Agent | Model | What the model does | Fallback |
|---|---|---|---|
| Agent 1 | Gemini `gemini-3.5-flash` | Reads the resume: contact info, skills, education, jobs, projects | Rule-based parser |
| Agent 3 | Groq `qwen/qwen3.8-27b` | Key points per job: what the employer wants, fit, gaps, interview topics | Rule-based key points |
| Agent 3 | local `all-MiniLM-L6-v2` (sentence-transformers) | Embeddings for the text-similarity part of the score | TF-IDF |

**No agent frameworks.** LangChain, LangGraph, CrewAI and similar aren't
used. All orchestration is our own Python code: `agents/pipeline.py` calls
Agent 1, then Agent 3, and every LLM call goes through our own small helper
(`agents/llm.py`) using plain HTTP requests.

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
  one-line change. Agent 1 uses `{"provider": "gemini", "model": "gemini-3.5-flash"}`,
  Agent 3 uses `{"provider": "groq", "model": "qwen/qwen3.8-27b"}`.
- **Validated JSON:** pass a pydantic model as `schema` and the reply is
  parsed as JSON and checked against it. Any problem (missing key, network
  error, bad JSON, wrong fields) raises one error type, `LLMError`, so the
  agent can fall back cleanly.
- **Retries:** "busy" API errors (HTTP 429/5xx) are retried up to 2 times.
  On a rate limit the API says how long to wait (`retry-after` header), and
  the helper waits that long, up to 30 seconds.

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
   It also asks for the candidate's **projects**, each with a name, role,
   date, a one-sentence summary and the skills used.
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
   - **Projects** from the Projects section. A project starts at a title
     line such as `Car Rental System | Lead Developer Jun 2024`: the part
     before `|` is the name, the part after is the role, and the date is
     pulled out. The summary is the project's first bullet (wrapped lines
     are joined), and the skills are the `skills.txt` keywords found in
     that project's text.

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
all four. After projects were added, both parsers also found the same
projects with the same name, role, date and skills on all four. Only the
summaries differ: the rules copy the first bullet, while the LLM writes its
own sentence covering the whole project. The rule-based parser is already tuned for these formats, so the
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
  "projects": [{"name": "Sales Trend Dashboard", "role": "Team Lead", "date": "Mar 2022",
                "summary": "Built an interactive Tableau dashboard ...",
                "skills": ["SQL", "Tableau", "pandas"]}],
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
resume profile fits each job, ranks them, and uses an LLM to pull out the
**key points** of each job: what the employer is actually looking for, how
the candidate fits, the gaps, and what the interview agent should ask about.

**Run it**
```bash
python agents/job_synthesizer.py                                   # sample.pdf + fake jobs
python agents/job_synthesizer.py path/to/resume.pdf path/to/jobs.json
python agents/job_synthesizer.py --rules                           # no LLM calls at all
```
Prints a ranking table followed by the key points of every job, and saves
everything to `output/ranked_jobs.json`. The full run (Gemini parse,
embeddings, one Groq call per job) took about 1.5 minutes for 6 jobs,
mostly spent waiting on Groq's free-tier rate limit (see Known limitations).

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
  "score": 84.0,
  "matched_skills": ["Java", "C++", "Git", "GitHub", "Data Structures", "Algorithms"],
  "missing_skills": ["Linux"],
  "reason": "Matches 6/7 skills; missing Linux.",
  "breakdown": {"skill_points": 51.4, "text_similarity": 0.494, "text_points": 32.5,
                "text_method": "embeddings",
                "required_years": 0, "candidate_years": 2.6, "experience_penalty": 0,
                "required_degree": "Bachelor", "candidate_degree": "Bachelor",
                "degree_penalty": 0},
  "assessment": {
    "key_points": ["Ability to work with performance-critical C++ services",
                   "Strong foundational knowledge of Algorithms and Data Structures", "..."],
    "required_skills": ["C++", "Java", "Linux", "Algorithms", "Data Structures"],
    "nice_to_have": ["Git", "GitHub"],
    "fit_summary": "The candidate is a strong fit due to their relevant CS degree, ...",
    "gaps": ["Linux"],
    "interview_focus": ["Probing the candidate's experience with Linux environments, ...", "..."]
  },
  "assessed_by": "llm"
}
```

### Key points (LLM, `assess_job`)

For each job, Agent 3 sends Groq (`qwen/qwen3.8-27b`) the job posting, a
short candidate summary, and the score our code already computed. It gets
back validated JSON:

| Field | Meaning |
|---|---|
| `key_points` | 1–8 points on what the employer is actually looking for |
| `required_skills` | what the posting says is required |
| `nice_to_have` | what the posting calls a plus / preferred |
| `fit_summary` | 1–2 sentences on how well the candidate fits |
| `gaps` | what the candidate is missing for this job |
| `interview_focus` | 1–6 topics **the interview agent should ask about** |

- **The LLM does not change the score.** The score is computed by code
  before the LLM is called. The schema has no score field, so even if the
  model sends one, it's dropped.
- **Privacy:** the candidate summary includes skills, education, job titles
  and projects, but not the name, email, phone or links.
- **Fallback:** if Groq fails (missing key, rate limit after retries, bad
  JSON, missing fields), the same six fields are filled from the rule-based
  requirements: key points list the required skills, years and degree,
  `fit_summary` is the one-line `reason`, `gaps` are the missing skills
  plus any experience/degree gap, and `interview_focus` lists the missing
  skills first, then the matched ones. `assessed_by` says which was used:
  `"llm"` or `"rules"`.
- **Why Qwen:** we compared the Groq models available in Nov 2026
  (`gpt-oss-120b`, `gpt-oss-20b`, `qwen3.8-27b`) on the same jobs. Qwen
  stayed closest to the postings (the others invented "nice to have"
  skills or misread the candidate's degree status), used the candidate's
  projects in its interview topics, and was the fastest (~1 s per job).

### Example run: `data/sample.pdf` + `tests/fixtures/jobs.json`

Resume parsed by Gemini, text similarity from embeddings, key points by
Qwen for all 6 jobs:

| # | Score | Job | Key points (what they want) | Gaps | Interview focus (first topics) |
|---|---|---|---|---|---|
| 1 | 100.0 | Junior Java Developer | Clean OOP code, data structures & algorithms, Java, Git, CS background | none | Car Rental System's OOP design; data-structure depth; Git workflow |
| 2 | 84.0 | Software Engineering Intern | Performance-critical C++, algorithms, Linux, Java tooling, current CS student | Linux | Linux experience; C++ performance practices; algorithm problem-solving |
| 3 | 39.1 | Frontend Web Developer | Responsive sites for small businesses, React + TypeScript, working with designers, Git | JavaScript, TypeScript, React | Any JS/React work; Java/OOP → TypeScript; responsive design |
| 4 | 11.3 | Data Science Intern | Python + pandas, SQL, basic ML with scikit-learn, stats/CS background | Python, SQL, Tableau, ML, pandas, scikit-learn | Motivation to switch to data science; any Python/SQL self-study |
| 5 | 9.3 | Senior Backend Engineer | Lead high-traffic payment systems, mentoring, Java/SQL/Docker/AWS, 5+ years | 5+ years, SQL, Docker, AWS, REST API, mentoring | Real backend experience; payment-system design; SQL/Docker exposure |
| 6 | 6.9 | Machine Learning Engineer | Deep learning for NLP/vision, Python, PyTorch/TensorFlow, Docker on AWS, Master's | Python, PyTorch/TensorFlow, Docker/AWS, ML background, Master's | Motivation for ML; steps taken to learn Python/PyTorch; the Master's requirement |

### How scoring works

Each job gets a score from 0 to 100:

| Part | Points | How |
|---|---|---|
| Skill overlap | up to **60** | 60 × (matched skills ÷ skills the job lists) |
| Text similarity | up to **40** | embedding cosine similarity, mapped to points (see below) |
| Experience gap | **−10 per year** short, max −30 | if your total years < the job's minimum |
| Degree gap | **−5** | if the job asks for a higher degree than you have |

The final score is kept between 0 and 100.

**Why these weights?** Skills are the most direct signal (recruiters screen
on "do you have the listed skills"), so they count most. Text similarity
catches overlap the skill list misses, like job titles, field of study, or
related wording, but it's noisier, so it counts less. The experience penalty
pushes senior roles down the list without removing them. The degree penalty
is small because students often apply while still finishing a degree.

**Text similarity with embeddings** (`agents/embeddings.py`). An embedding
model turns a text into a vector that captures its meaning, so related
wording ("developer" / "engineer") lands close together even when the words
differ. Agent 3 uses a local sentence-transformers model
(`all-MiniLM-L6-v2`, set in `EMBED_CONFIG`). It runs on your computer, so
no API key is needed and no resume text is sent anywhere. The model is
downloaded once (about 90 MB) on first use.

- The model reads at most 256 tokens (about 180 words), and a resume is
  longer. So the resume is split into 120-word chunks, each chunk is
  embedded, and a job's similarity is that of its **best-matching chunk**.
- Similarity is mapped to points on a straight line between two anchors:
  **0.25 or less → 0 points, 0.55 or more → 40 points** (0.40 → 20).
- The anchors were measured with `python scripts/calibrate_embeddings.py`:
  our 4 resumes scored 0.01–0.28 against unrelated postings (pastry chef,
  nurse, truck driver) and about 0.50–0.63 against the postings that
  clearly fit them.
- **Scores are absolute.** A job's score depends only on the resume and
  that job. It's the same whether the job is scored alone or in a list of
  100, and adding or removing other postings doesn't change it.

**TF-IDF fallback.** If the embedding model can't be loaded (library not
installed, no internet on first use), Agent 3 prints a warning and uses
TF-IDF word overlap instead. TF-IDF values are small and not comparable
between runs, so they're rescaled: the most similar job **in that list**
gets the full 40 points. In this mode scores are relative to the other jobs.
`breakdown.text_method` says which was used: `"embeddings"` or `"tfidf"`.

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
By default (`use_llm=True`) Agent 1 parses with Gemini and Agent 3 writes key
points with Groq, each with its own fallback. `use_llm=False` makes no API
calls at all. The API always uses the default.

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
| `tests/test_llm.py` | `ask_llm`: Gemini/Groq requests, JSON + schema validation, missing keys, retries and `retry-after` |
| `tests/test_parser.py` | Agent 1 rule-based parser on 3 fake resumes with different heading, date and skill styles |
| `tests/test_parser_llm.py` | Agent 1 LLM parser: same format as rules, cleanup, every fallback case, pipeline default |
| `tests/test_job_synthesizer.py` | Requirement extraction, scoring, penalties, ranking (on the TF-IDF fallback), LLM key points with a fake Groq: score unchanged, no contact details in the prompt, every fallback case |
| `tests/test_embeddings.py` | Chunking, best-chunk similarity, the 0–40 mapping, scores not depending on other jobs, fallback |
| `tests/test_pipeline.py` | `run_pipeline` and the `/rank` endpoint (FastAPI `TestClient`) |

**Tests never call a real LLM or load the real embedding model.**
`tests/conftest.py` blocks the network function in `agents/llm.py` and the
model loader in `agents/embeddings.py` for every test. Tests that need a
reply or an embedding install a fake one. The suite runs offline, without
API keys, and downloads nothing.

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
- **Projects assume a layout**: a title line followed by bullets. A title
  with no `|`, no date and no bullet under it is missed, and a wrapped
  bullet line that contains a year can be mistaken for a new project. The
  summary is just the first bullet.

**Agent 1, both paths**
- **No OCR.** Scanned or image-only PDFs give no text.
- Overlapping jobs are added together, and year-only dates are approximated
  (start = January, end = December).

**Agent 3, key points (LLM)**
- **Groq free-tier rate limit:** `qwen/qwen3.8-27b` allows 1,000 output
  tokens per minute, about 3 jobs. Larger job lists wait on retries (6 jobs
  took ~1.5 minutes), and if the wait would exceed the retry budget, the
  remaining jobs fall back to rule-based key points. Check `assessed_by`.
- **The LLM can misjudge the candidate.** In our run it said a student
  still studying "possesses the required Bachelor's degree", and it counted
  total work years (cashier, test lead) toward "1+ years building websites".
  Key points are guidance for the interview agent, not facts. The score
  doesn't depend on them.
- **Output can vary between runs**, and job postings could contain
  instructions aimed at the model (prompt injection), the same as for Agent 1.
- **Job text is sent to Groq**, and so is a summary of the candidate
  without contact details.

**Agent 3, scoring**
- **Job requirement parsing is regex-based**: `min_years` takes the largest
  "N years" in the text, even if it isn't about experience (e.g. "founded
  10 years ago"). "MS"/"M.S." only counts as a Master's in a degree context
  ("MS in …", "M.S. degree", "Degree: MS", "BS/MS"), so "MS Excel" is ignored,
  but an unusual phrasing like "MS required" is missed.
- **Embedding similarity is noisy.** The anchors (0.25 / 0.55) come from 4
  resumes and 9 postings and only fit this model. Very short resumes give
  weak signals (a web developer's resume scored only 0.36 against a
  frontend job), and longer resumes have more chunks, so slightly more
  chances of a high best-chunk score.
- **Embeddings are slow to start and heavy to install**: loading the model
  takes several seconds per run, and it needs PyTorch.
- **The TF-IDF fallback is bag-of-words** and its scores are relative to
  the job list (see above).
- **The API has no authentication** and is meant for local use and demos.
