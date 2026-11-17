# INTERVAI

A multi-agent AI mock interview platform. This README covers **Agent 1
(Resume Parser)**, **Agent 2 (Job Search)**, **Agent 3 (Job Description Synthesizer + Match Scoring)**
the **pipeline / API**, and the **orchestrator** that runs them all and hands the chosen job to Agent 4.

| Agent | Kind of AI | Model | What the model does | Fallback |
|---|---|---|---|---|
| Agent 1 | **NLP: named-entity recognition** (supervised, trained by us) | spaCy `en_core_web_sm`, fine-tuned on 48 annotated resumes → `models/resume_ner/` | Finds names, companies, schools, degrees, job titles, skills and dates in the resume | Rule-based parser |
| Agent 2 | **LLM agent** (reason → act → observe loop with a search tool) | Groq `openai/gpt-oss-120b` | Decides which job searches to run and searches again if results are weak | Rule-based query from the profile |
| Agent 3 | **Semantic embeddings / semantic search** (no LLM) | local `all-MiniLM-L6-v2` (sentence-transformers) + KeyBERT | (1) Text-similarity part of the match score; (2) key points per job: KeyBERT picks the posting's key phrases, embedding search checks each against the resume, giving fit, gaps and interview topics | TF-IDF (score), rule-based key points |

Each agent uses a different kind of AI. **Agent 2 is the only LLM agent.**
Elsewhere, LLMs are kept only as optional comparison baselines, both **off
by default**: Agent 1 can parse with Gemini (`parser="llm"`), and Agent 3
can write key points with Groq (`rank_jobs(..., use_llm=True)`). By default
Agents 1 and 3 run entirely on this computer, with no API keys.

**No agent frameworks.** LangChain, LangGraph, CrewAI and similar aren't
used. All orchestration is our own Python code: `agents/orchestrator.py`
runs Agent 1 → Agent 2 → Agent 3 → the user's job choice → the Agent 4
handoff, `agents/pipeline.py` calls Agent 1, then Agent 3, `POST /search`
calls Agent 1, then Agent 2, and every
LLM call goes through our own small helper (`agents/llm.py`) using plain
HTTP requests.

## Setup

```bash
conda activate intervai_env
pip install -r requirements.txt
cp .env.example .env        # then put your API keys in .env (never commit it)
```

The empty template in `.env.example` lists three keys. Fill in the ones you
need:

| Key | Used by | Needed? |
|---|---|---|
| `GEMINI_API_KEY` | Agent 1's optional LLM parser (`parser="llm"`) | Only for that comparison |
| `GROQ_API_KEY` | Agent 2 (search agent); Agent 3's optional LLM key points | For `/search` (without it Agent 2 falls back to one rule-based search) |
| `RAPIDAPI_KEY` | Agent 2's job source (JSearch) | For `/search` |

`/rank` and `run_pipeline()` need no keys by default.

For `RAPIDAPI_KEY`, create a RapidAPI account, subscribe to the free BASIC
plan of the JSearch API (200 requests a month, no credit card), and copy
the key from the API's Code Snippets panel. Write it in `.env` as
`RAPIDAPI_KEY=your_key`, with no quotes. Without it, Agent 2 can't fetch
jobs and `/search` returns an error.

Run every command below from the project root.

### LLM setup (`agents/llm.py`)

Every LLM call (Agent 2, and the optional LLM modes of Agents 1 and 3) goes through one shared helper,
`ask_llm(prompt, provider, model, schema=None)`:

- **Providers:** `"gemini"` (key `GEMINI_API_KEY`) and `"groq"` (key `GROQ_API_KEY`),
  read from `.env` with python-dotenv. Keys are never printed or committed.
- **Per-agent model:** each agent sets its provider and model in one
  `LLM_CONFIG` dict at the top of its file, so switching models is a
  one-line change. Agent 2 uses `{"provider": "groq", "model": "openai/gpt-oss-120b"}`.
  Agent 3's optional key points use `{"provider": "groq", "model": "qwen/qwen3.8-27b"}`.
  Agent 1's optional LLM parser uses `{"provider": "gemini", "model": "gemini-3.5-flash"}`.
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
Agent 1 is an **NLP named-entity-recognition (NER) agent**: a spaCy model
we trained ourselves on annotated resumes. It has a rule-based fallback,
plus an optional Gemini parser for comparison. All three return exactly the
same profile format, marked with `"parsed_by": "ner" | "llm" | "rules"`.

### NER path — `parse_resume_ner(path)` (default)

**What NER is:** a model that reads text and marks spans of it as
entities of given types. Ours recognizes six types:

| Label | Example |
|---|---|
| `PERSON` | `Priya Nandakumar` |
| `ORG` | `Brightline Analytics`, `Northfield State University` |
| `DEGREE` | `Bachelor of Science in Statistics` |
| `JOB_TITLE` | `Junior Data Analyst` |
| `SKILL` | `Python`, `REST API`, `Kubernetes` |
| `DATE` | `Jan 2022`, `05/2024`, `2021`, `Present` |

**How a resume becomes a profile:**
1. Read the text (same readers as the rule-based path).
2. Run the trained model to get the entities.
3. Assemble them by layout:
   - name = first `PERSON`, unless it is missing, looks like an organisation ("Web Studio": contains Studio, University, Inc, …) or is only part of the first line (a hyphenated name cut in two); then the resume's first non-empty line if it looks like a name (2–4 capitalised words, no digits, `@` or `|`). An organisation is never used as the name
   - skills, by **entity linking**: each `SKILL` entity is mapped to the
     canonical `skills.txt` names it contains ("Java Programming" → Java,
     "Microsoft Excel" → Excel, "Object-Oriented Programming" → OOP; other
     spellings live in `data/skill_aliases.txt`). Entities that link to no
     known skill ("Kushtia", a place) are dropped, and duplicates removed.
     Then every `skills.txt` skill found anywhere else in the text
     (coursework, projects) is added, so "Analysis of Algorithms" gives
     Algorithms. `skill_sources` records `"ner"` or `"text"` for each skill.
   - **planned work isn't a skill:** a skill that only appears after "future",
     "roadmap", "plan(s/ned) to", "to learn" or "learning" in its sentence
     ("documented a future roadmap for GUI development") is ignored.
     "Machine/Deep Learning" are skill names, not plans.
   - education = first `DEGREE`, the `ORG` up to 2 lines above or 1 below it, and the nearest date
   - jobs = one per line with a `JOB_TITLE`, with the `ORG` on the same or a neighbouring line and the next two real dates (job titles inside Projects, Leadership or Awards sections are ignored). If the model finds a title but no company, the title's line is split like the rule-based parser does, first part = company, second = title ("Chick-fil-A | Team Member (Cashier)"). This also fixes lines where the model tagged both halves as titles ("Quillstone Media | Web Developer")
4. **Python does the date math**, exactly as in the other parsers. `DATE`
   entities that aren't real dates (the model sometimes tags "weekly" or
   "three years") are skipped.
5. Email, phone and links use regex (fixed patterns, no learning needed).
   Projects use the rule-based project reader (projects aren't an NER label).

**Fallback:** if the model can't be loaded (not trained yet, spaCy
missing) or crashes, Agent 1 logs a warning and uses the rule-based parser
(`"parsed_by": "rules"`).

### Training data and training (`tests/fixtures/ner/`, `scripts/train_ner.py`)

- **Data:** 60 FAKE resumes from `tests/fixtures/ner/make_ner_data.py`
  (fixed seed). They vary in 5 header styles, 3 heading styles, 7 job
  layouts, 5 education layouts, 5 date formats, 4 skills-section styles,
  and optional summary, projects, coursework, honors, spoken languages
  and leadership sections. Every skill in `data/skills.txt` appears, plus
  skills that aren't in it. 1,696 entities in total.
- **Annotation format:** one JSON object per line in `resumes.jsonl`:
  ```json
  {"id": "resume-07", "text": "Maya Okafor\n...",
   "entities": [[0, 11, "PERSON", "Maya Okafor"], [61, 81, "ORG", "Brightline Analytics"]]}
  ```
  Each entity is `[start, end, label, text]` with character offsets
  (`text[start:end]` is the entity). The generator records the offsets
  while writing the text, so labels can't drift. A check makes sure every
  span starts and ends on a spaCy token boundary.
- **Labeling rules:** `JOB_TITLE` only in work experience. Project roles,
  club roles, course names, honors, spoken languages, cities and links are
  **not** labeled, so the model learns to leave them alone. Each date is
  its own `DATE` (a range is two).
- **Training:** start from `en_core_web_sm` (already knows PERSON/ORG/DATE),
  add the new labels, and fine-tune only the NER component: 30 epochs,
  batch size 4, dropout 0.3, fixed seed. **48 resumes train, 12 test**
  (80/20). Takes about 20 seconds.
  ```bash
  python scripts/train_ner.py     # retrains and saves models/resume_ner/ (15 MB, committed to git)
  ```

### Evaluation (exact match: start, end and label must all be right)

| Label | Synthetic test (12 resumes) P / R / F1 | Hand-written (3 resumes, unseen layouts) P / R / F1 |
|---|---|---|
| PERSON | 1.000 / 0.917 / 0.957 | 0.667 / 0.667 / 0.667 |
| ORG | 1.000 / 1.000 / 1.000 | 1.000 / 0.889 / 0.941 |
| DEGREE | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| JOB_TITLE | 1.000 / 1.000 / 1.000 | 0.750 / 1.000 / 0.857 |
| SKILL | 0.995 / 1.000 / 0.997 | 0.966 / 1.000 / 0.982 |
| DATE | 1.000 / 1.000 / 1.000 | 0.900 / 1.000 / 0.947 |
| **Overall** | **0.997 / 0.997 / 0.997** (332 entities) | **0.915 / 0.970 / 0.942** (67 entities) |
| *Untrained `en_core_web_sm`* | *0.242 / 0.145 / 0.181* | *0.333 / 0.134 / 0.191* |

- The **synthetic** score is optimistic: those resumes come from the same
  generator as the training data. The **hand-written** set (the 3 fixture
  resumes, annotated in `tests/fixtures/ner/handmade_eval.jsonl`) uses
  layouts the generator never produces, so it's the fairer number.
- Typical errors: a hyphenated surname split into two entities
  ("Elena Vasquez" + "Moreau" as a SKILL), a company tagged as a job title
  in an unfamiliar position, and words like "weekly" tagged as DATE.

### LLM path — `parse_resume_llm(path)` (optional, `parser="llm"`)

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

This was Agent 1's default before the NER agent. It's kept as a
comparison point and as an option.

**Fallback:** if the key is missing, the API is down or busy, the reply
isn't valid JSON, or it fails validation (e.g. a job that ends before it
starts), Agent 1 logs a warning and returns the rule-based result instead,
with `"parsed_by": "rules"`. The pipeline never breaks because of the LLM.

### Rule-based path — `parse_resume(path)` (fallback for both)

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
python agents/resume_parser.py                      # NER, uses data/sample.pdf
python agents/resume_parser.py path/to/resume.docx  # NER, your file
python agents/resume_parser.py --llm                # Gemini instead
python agents/resume_parser.py --rules              # rule-based only
```
Prints the profile and saves it to `output/profile.json`.

**Compare the three parsers**
```bash
python scripts/compare_parsers.py      # 3 fake fixtures + data/sample.pdf (if present)
```
Runs rules, NER and Gemini on each resume and prints only the fields where
they disagree, then how many of the 8 compared fields (name, email, phone,
skills, education, jobs, years, projects) each pair agrees on. Our run:

| Resume | Differences |
|---|---|
| `resume_title_case.docx` | Gemini misses **Excel** (only in a job bullet); rules and NER find it |
| `resume_numeric_dates.docx` | all three agree on every field |
| `resume_comma_skills.docx` | NER splits the hyphenated name ("Elena Vasquez" + "Moreau" as a skill) and puts the company in the title slot for the first job |
| `data/sample.pdf` | (before entity linking and company recovery) NER missed the company "Chick-fil-A", added a place name and a course name as skills, and kept "Microsoft Excel" whole; Gemini adds "Word"; jobs, months and education agree |

| Pair | Fields that agree (of 32) |
|---|---|
| rules vs NER | 27 |
| rules vs Gemini | 30 |
| NER vs Gemini | 26 |

**Effect on Agent 3** (sample.pdf, same jobs): with the NER profile the two
best matches score **90.0 and 75.4** instead of **100.0 and 84.0**, because
NER doesn't count "Algorithms", which appears only as a course name. The
order of all six jobs is the same.

Earlier, Gemini and the rules agreed on everything except Excel/Word, with
the same projects on all four resumes (only summaries differ, because Gemini
writes its own sentence).

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
  "parsed_by": "ner",
  "raw_text": "full resume text ..."
}
```
`parsed_by` is `"ner"`, `"llm"` or `"rules"`, so you can always tell which path produced a profile.
`raw_text` is saved in the file (Agent 3 needs it) but left out of the
printout and out of the pipeline/API result.

---

## Agent 2 — Job Search Agent (`agents/job_search_agent.py`)

Finds job postings that fit a parsed profile. Agent 2 is an **LLM agent
(Groq) with a rule-based fallback**. It returns jobs in the exact format
Agent 3 and `run_pipeline()` expect: `{id, title, company, location, url,
description}`, all strings.

| File | Role |
|---|---|
| `agents/job_search_agent.py` | The AI agent: decides which searches to run and when to stop |
| `agents/job_search.py` | Non-AI toolkit the agent calls: job format, provider interface, cleaning, dedupe, rule-based `build_query` |
| `agents/jsearch_provider.py` | Real job source: JSearch on RapidAPI (`/search-v2`) |

### How the agent works

`run_job_agent(profile, provider, location="", limit=20, max_steps=4)` runs
our own loop (no agent frameworks):

1. **Reason.** The LLM gets a summary of the profile (skills, recent titles,
   years, projects) and the searches so far, and replies with JSON:
   `search` with a query, or `finish`.
2. **Act.** The loop cleans the query (dedupes and caps keywords, keeps the
   caller's location, fixes a bad experience level), skips repeats, and runs
   `search_jobs()` as the tool.
3. **Observe.** The LLM is shown a short summary of the results (counts and
   up to 5 job titles with companies, never full descriptions) and decides
   whether to search again. If results are weak it tries a different query.
4. It stops when the LLM finishes, the job limit is reached, or after
   `max_steps` steps (default and maximum `MAX_STEPS` = 4). Each step is at
   most one search, so `max_steps` also caps the JSearch requests per run.
   Jobs from all searches are merged without duplicates.

**The LLM never sees contact details or resume text**: only skills, job
titles, years of experience and project names and skills.

**Fallback:** if the LLM fails at the first step (missing key, API down, bad
JSON) or finishes without searching, Agent 2 runs one search with the
rule-based `build_query(profile)`. The result's `mode` is `"llm"` or
`"rules"`, so you can tell which path ran.

**Search records and errors.** Every step that searched has a `"search"`
record in the trace: `query_text`, `location`, `http_status`, `raw_count`,
`returned` (after cleaning), `from_cache`, and `error` if it failed. A failed
search (401/403 key, 429 quota, 500, network) is logged as a WARNING with its
status code and recorded as `search_error`; it is never reported as "no jobs".

**Widening.** A small city often has no postings for a title + skills query
(on Nov 16, `Software Engineer Java OOP in Monroe, LA` returned HTTP 200 with
0 jobs). If every search comes back empty without an error, Agent 2 widens
the area by itself: city → state → Remote (`Monroe, LA` → `Louisiana` →
`Remote`; the state is spelled out because "LA" also means Los Angeles),
searching the job title only, and stops at the first area with jobs. Each
widening is a `{"step": "widen", "widened_from", "widened_to", ...}` trace
entry. That is at most 2 extra requests on top of `max_steps`;
`run_job_agent(..., widen=False)` turns it off. There is no widening after an
error (a bad key or a used-up quota would only fail again) or when the
location is already Remote. After an empty search the LLM is also told to try
the job title alone.

### Job source: JSearch

`JSearchProvider` sends one request per search to JSearch's `/search-v2`
endpoint and returns up to 10 jobs. Setup:

1. Subscribe to the free BASIC plan of JSearch on RapidAPI (200 requests a
   month, no credit card).
2. Put your key in `.env` as `RAPIDAPI_KEY=...` (see `.env.example`).

The query text is the job title plus at most 2 keywords plus the location
(for example `Software Developer Java Python in Baltimore, MD`), because
longer queries returned no results in our tests. Failures (missing key,
rejected key, quota reached, HTTP or network errors, bad JSON) raise one
error type, `ProviderError`, and messages never contain the key. Other
sources can be added by subclassing `JobProvider`.

**Query cache.** To save the monthly quota, results are cached on disk in
`output/jsearch_cache.json` (git-ignored) for **24 hours**, keyed by the
final query text above. Repeating a search within a day (for example
re-running the same resume) makes no request. Only successful, non-empty
results are cached: errors and empty results are always asked for again. A
corrupt cache file is ignored and rewritten, never fatal. **Delete
`output/jsearch_cache.json` to get fresh results.** `JSearchProvider(cache_path=None)`
turns the cache off.

**What each request did.** After every search, `provider.last_request` holds
`{"query_text", "http_status", "raw_count", "from_cache"}`. `raw_count` is
the number of jobs in JSearch's reply (`data -> jobs`); an error leaves it
`None`, so a failure never looks like "0 jobs". `ProviderError.status` is the
HTTP status code (`None` for network errors). A real JSearch reply looks like
`{"status": "OK", "request_id", "parameters", "data": {"jobs": [...], "cursor"}}`.

**Result format**
```python
result = run_job_agent(profile, JSearchProvider(), location="Baltimore, MD")
result["jobs"]    # [{id, title, company, location, url, description}, ...]
result["trace"]   # each step: thought, action, query, returned, new
result["mode"]    # "llm" or "rules"
```

### Run it (via the API)

```bash
uvicorn api:app --reload
curl -X POST http://127.0.0.1:8000/search \
     -F "resume=@data/sample.pdf" \
     -F "location=Baltimore, MD" \
     -F "max_steps=2"
```
`POST /search` takes a `.pdf`/`.docx` resume, an optional `location`, a
`limit` (1 to 50, default 20) and `max_steps` (1 to 4, default 4; lower it
to use fewer JSearch requests). Agent 1 parses the resume with the NER model
(`parse_resume_ner`, rule-based fallback), then Agent 2 searches. It returns
`{"profile", "jobs", "search_mode", "trace"}`. The `jobs` list can be passed
straight to `/rank`. HTTP 400 means a bad file type, `limit` or `max_steps`.
HTTP 502 means the job source failed (check `RAPIDAPI_KEY` and the
monthly quota). If a search finds nothing, `jobs` is an empty list, and
`/rank` rejects an empty list, so check it first.

---

## Agent 3 — Job Synthesizer + Match Scoring (`agents/job_synthesizer.py`)

Reads job postings, works out what each one requires, scores how well a
resume profile fits each job, ranks them, and pulls out the **key points**
of each job: what the employer is looking for, how the candidate fits, the
gaps, and what the interview agent should ask about.

**Kind of AI: semantic embeddings / semantic search.** Both the score and
the key points come from one local sentence-transformers model
(`all-MiniLM-L6-v2`). No LLM and no API key in the default path.

**Run it**
```bash
python agents/job_synthesizer.py                                   # sample.pdf + fake jobs
python agents/job_synthesizer.py path/to/resume.pdf path/to/jobs.json
python agents/job_synthesizer.py --rules                           # rule-based resume parser
python agents/job_synthesizer.py --llm                             # optional Groq key points
```
Prints a ranking table followed by the key points of every job, and saves
everything to `output/ranked_jobs.json`. The default run (NER parse,
embeddings, KeyBERT) takes about 3 seconds for 6 jobs once the model is
loaded. `pip install -r requirements.txt` installs everything; the
embedding model (about 90 MB) downloads automatically the first time it's
used, then works offline.

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
  "score": 67.0,
  "matched_skills": ["Java", "C++", "Git", "Linux"],
  "missing_skills": ["GitHub", "Data Structures", "Algorithms"],
  "reason": "Matches 4/7 skills; missing GitHub, Data Structures, Algorithms.",
  "breakdown": {"skill_points": 34.3, "text_similarity": 0.495, "text_points": 32.7,
                "text_method": "embeddings", "...": "..."},
  "assessment": {
    "key_points": ["Summer internship", "Computer Science", "performance-critical C++ services",
                   "GitHub", "Bachelor", "Linux"],
    "required_skills": ["Java", "C++", "Linux", "Data Structures", "Algorithms"],
    "nice_to_have": ["Git", "GitHub"],
    "fit_summary": "Matches 4/7 skills; missing GitHub, Data Structures, Algorithms. The resume shows 5 of 6 key points of the posting.",
    "gaps": ["Data Structures", "Algorithms", "Summer internship"],
    "interview_focus": ["Data Structures (asked for, not on the resume)", "...",
                        "performance-critical C++ services (ask for an example; resume: skills: C++)"],
    "key_point_matches": [
      {"phrase": "Summer internship", "relevance": 0.664, "resume_similarity": 0.41,
       "covered": false, "evidence": null},
      {"phrase": "Computer Science", "relevance": 0.51, "resume_similarity": 0.669,
       "covered": true, "evidence": "Bachelor of Science in Computer Engineering"},
      "..."
    ]
  },
  "assessed_by": "embeddings"
}
```
(Fake resume `tests/fixtures/resume_numeric_dates.docx`. `assessed_by` is
`"embeddings"` by default, `"rules"` if the embedding model is unavailable,
or `"llm"` when the optional Groq key points are on.)

### Key points (KeyBERT + embeddings, default — `agents/keypoints.py`)

1. **Candidate phrases.** spaCy's general English model (`en_core_web_sm`,
   the same pinned package Agent 1 builds on) finds the noun phrases in the
   posting ("OOP principles", "high-traffic payment services"). Leading or
   trailing filler words ("strong", "knowledge", "requirements") are cut.
2. **KeyBERT picks the key phrases.** It embeds the whole posting and every
   candidate phrase with Agent 3's own model, and keeps the 6 phrases
   closest in meaning to the posting. MMR (diversity 0.5) keeps them
   different from each other, so you don't get five versions of
   "internship". KeyBERT gets our `embeddings.embed()` as its model, so the
   model is loaded only once.
3. **What kind of phrase is it?** (`phrase_kind`) — checked in order:
   **skill** (contains a `skills.txt` skill: "performance-critical C++
   services"), **degree** (is a degree, or follows a degree word in its
   sentence: "Bachelor's degree in *Computer Science*", "students studying
   *Computer Science*"), **experience** (follows "N years" in its sentence:
   "5+ years of *professional backend* experience"), otherwise **role**: it
   describes the role, team or company ("junior developer", "Summer
   internship", "designers", "small business clients"). Role phrases stay in
   `key_points` but are never gaps or "asked for, not on the resume".
4. **Is it on the resume?** (`coverage`)
   - *skill*: every skill in the phrase is one of the candidate's skills (exact match)
   - *degree*: a level ("Bachelor") needs a degree at least that high; a
     field ("Computer Science") must appear word for word in the candidate's
     degree, unless the posting accepts "a related field"
   - *experience / role*: semantic search. The phrase is embedded and
     compared with every resume line; the best line is **evidence** if its
     similarity is ≥ **0.45**, or ≥ **0.55** when a multi-word phrase shares
     only one word with it ("customer data" ↔ "…multi-customer support").
     Wrapped lines are joined back to their bullet first, and lines under 4
     words (headings like "Work Experience") are never evidence.
   Skills and degrees use rules because similarity got facts wrong: it rated
   "Computer Science" ↔ "B.S. Computer *Engineering*" (0.67) above
   "Statistics" ↔ "B.S. in Statistics" (0.61).
5. **The fields**, built in Python from the phrases, the skill list and the matches:

| Field | How |
|---|---|
| `key_points` | the KeyBERT phrases, most relevant first. A phrase that is nothing but skills already listed in `nice_to_have` or `gaps` ("GitHub", "Linux") is dropped, so **each skill appears in only one of key_points / nice_to_have / gaps**; KeyBERT returns 4 spare phrases so 6 remain |
| `required_skills` | the posting's skills from `data/skills.txt`, minus the nice-to-haves |
| `nice_to_have` | skills in a sentence with "plus", "bonus", "preferred", "desired", … |
| `fit_summary` | the one-line `reason` + "the resume shows K of N key points" |
| `gaps` | missing required skills, skill/degree/experience key phrases not on the resume, years/degree gaps (never role phrases) |
| `interview_focus` | up to 3 gaps to probe first, then covered key points to ask for an example of, with the resume evidence |
| `key_point_matches` | every phrase with its kind, relevance, resume similarity, covered and evidence |

- **Why 0.45 / 0.55?** Measured on the 3 fake fixture resumes × every
  non-skill phrase of the 6 fixture jobs (102 pairs). True matches:
  "dashboards" 0.54, "analytics" (a data-analyst job line) 0.53,
  "building websites" ← "Built client sites with React and Node.js" 0.47
  (no shared word at all). False matches: 0.44 and below ("junior
  engineers" ← a degree line, "datasets" ← an Excel line).
- **Skills from the whole resume.** Agent 3 matches against Agent 1's
  skills plus every `skills.txt` skill found anywhere in the resume text,
  so a skill mentioned only in coursework ("Analysis of Algorithms") or a
  project counts. `breakdown.matched_from_text` lists those.
- **The key points never change the score.** The score is computed first,
  by the scoring code.
- **Speed and privacy:** about 0.3 s per job, offline, nothing sent anywhere.
- **Fallback:** if the embedding model or KeyBERT can't be used, one
  warning is printed and every job gets the rule-based key points (the
  required skills, years and degree; `gaps` = missing skills + years/degree;
  `interview_focus` = missing skills, then matched ones).

### Optional: LLM key points (Groq, off by default)

The earlier LLM version is kept for comparison: `rank_jobs(profile, jobs,
use_llm=True)`, `run_pipeline(..., llm_key_points=True)` or
`python agents/job_synthesizer.py --llm`. It sends Groq (`qwen/qwen3.8-27b`)
the posting, a candidate summary without contact details (graduation stated
as "in progress" when it's in the future; total years labeled "related or
not"), and the score our code computed, and gets back the same fields as
validated JSON. Only the top 5 jobs (`LLM_CONFIG["top_n"]`) go to Groq;
the rest, and any job where Groq fails, get the embedding version. The LLM
never changes the score (the schema has no score field). Qwen was chosen
over `gpt-oss-120b` / `gpt-oss-20b` for staying closest to the postings.

### KeyBERT vs. Groq key points (`scripts/compare_key_points.py`)

Same fake resume (`resume_numeric_dates.docx`: Computer Engineering
graduate, C++/Java/Docker/Linux, robotics internship) and the 6 fixture
jobs. Run on 2026-11-16:

| Job | KeyBERT + embeddings: key points | Groq `qwen3.8-27b`: key points (shortened) |
|---|---|---|
| Junior Java Developer | junior developer, Java, Computer Science, internal tools, Git, clean code | Clean OOP code in Java; data structures & algorithms; Git; CS background, no experience needed |
| Software Engineering Intern | Summer internship, Computer Science, performance-critical C++ services, GitHub, Bachelor, Linux | Algorithms & data structures; performance-critical C++; Java tooling; current CS students; Linux |
| Data Science Intern | Tableau, analytics, dashboards, customer data, scikit-learn, Bachelor | Python + pandas data cleaning; SQL; basic ML with scikit-learn; Statistics/Data Science/CS background |
| Machine Learning Engineer | TensorFlow, Design and train Deep Learning models, Computer Vision products, AWS, Computer Science, Docker | Deep learning for NLP + vision; Python daily; PyTorch & TensorFlow; Docker on AWS; Master's required |
| Senior Backend Engineer | professional backend, AWS, high-traffic payment services, Java, Docker containers, REST API layer | Lead design of high-traffic payment services; mentor juniors / own architecture; Java + REST; PostgreSQL + SQL; containers on AWS |
| Frontend Web Developer | building websites, responsive websites, small business clients, designers, React, Git | Responsive sites for small businesses; HTML/CSS/JS/TS/React; Git on every project; working with designers; 1+ year building websites |

| | KeyBERT + embeddings | Groq LLM |
|---|---|---|
| Time for 6 jobs | **1.8 s** | 88 s (free-tier rate limit) |
| API calls / key needed | 0 / no | 6 / yes |
| Data leaves the computer | no | job text + candidate summary |
| Same answer every run | yes | no |
| Groq's 28 key points with a matching KeyBERT phrase (embedding similarity ≥ 0.5) | 16 / 28 (mean best similarity 0.54) | — |
| Match scores | identical | identical |

What the comparison shows:
- **Same topics, different form.** KeyBERT returns short phrases taken
  from the posting; Groq writes sentences. Most of Groq's points that
  KeyBERT "missed" are plain skill lists (Python, pandas, SQL, PostgreSQL),
  which Agent 3 already reports in `required_skills` and `gaps` from the
  skill list.
- **Groq reasons about the candidate; KeyBERT can't.** Groq noticed that the
  candidate already graduated (so a current-student internship doesn't fit),
  that Computer Engineering ≠ Computer Science, and that only 1.5 of the
  3 years were relevant, and it wrote interview questions about the
  candidate's own robotics project. KeyBERT counted "Computer Science" as
  covered because of the "Computer Engineering" line (similarity 0.67).
- **KeyBERT is more literal:** some picks are generic ("Bachelor",
  "Computer Science", "junior developer"), and descriptive phrases like
  "internal tools" become gaps when the resume doesn't mention them.
- **Why embeddings are the default anyway:** a different kind of AI from
  Agent 2's LLM (the course requirement), offline, free, deterministic,
  50× faster, and every key point carries its evidence.

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

run_pipeline("data/sample.pdf", jobs, parser="llm")             # Agent 1 uses Gemini instead
run_pipeline("data/sample.pdf", jobs, llm_key_points=True)      # Agent 3 key points from Groq
run_pipeline("data/sample.pdf", jobs, use_llm=False)            # forbid every API call
```
It checks the job list first and raises `ValueError` if a job is missing fields.
By default everything runs locally: Agent 1 parses with the NER model
(`parser="ner"`) and Agent 3 scores and writes key points with embeddings
(KeyBERT), each with its own fallback. No API calls are made unless you ask
for an LLM. `parser` can be `"ner"`, `"llm"` or `"rules"`. `use_llm=False`
forbids all API calls (an `"llm"` parser becomes `"rules"` and
`llm_key_points` is ignored). The API always uses the defaults.

**Web API** (`api.py`, FastAPI):
```bash
uvicorn api:app --reload
```
`POST /rank` takes a multipart form with `resume` (a `.pdf`/`.docx` file) and
`jobs` (a JSON list as text), and returns `{"profile": ..., "ranked_jobs": [...]}`.

`POST /search` (Agent 2) finds jobs for a resume, and its jobs result can be sent to `/rank`.
```bash
curl -X POST http://127.0.0.1:8000/rank \
     -F "resume=@data/sample.pdf" \
     -F "jobs=<tests/fixtures/jobs.json"
```
Interactive docs: http://127.0.0.1:8000/docs. Bad file types, invalid JSON
or incomplete jobs return HTTP 400. The uploaded resume goes to a temp file
that's deleted as soon as the request finishes.

---

## Orchestrator — the full INTERVAI run (`agents/orchestrator.py`)

Runs every agent in order, in our own code (no agent frameworks):

```
Agent 1 (NER) -> Agent 2 (LLM + JSearch) -> Agent 3 (embeddings) -> HITL: user picks a job -> Agent 4 handoff
   profile          job postings              ranked jobs              chosen job           interview_brief.json
```

**Run it**
```bash
python agents/orchestrator.py data/sample.pdf --location "Monroe, LA"   # real job search
python agents/orchestrator.py data/sample.pdf --offline                 # fake jobs, no JSearch/Groq calls
python agents/orchestrator.py data/sample.pdf --max-steps 1             # at most 1 JSearch request
python agents/orchestrator.py data/sample.pdf --offline --auto          # pick the top job, no questions
```
Use `--offline` during rehearsals: it uses the 6 fake jobs in
`tests/fixtures/jobs.json` on purpose and saves the JSearch quota (about 200
requests a month). Agent 2's own on-disk cache also avoids repeat requests.

**From Python (e.g. a dashboard):**
```python
from orchestrator import run_intervai, make_terminal_chooser
result = run_intervai("resume.pdf", "Remote", choose_job=my_callback, offline=False)
result["ranked_jobs"], result["selected"], result["handoff"]["brief"], result["trace_path"]
```
`choose_job(ranked_jobs)` gets Agent 3's ranking (best first) and returns one
of the ranked jobs, `{"custom_role": "Data Engineer"}` (a role that isn't in
the list; Agent 3 scores it from the role name alone), or `None` to stop
before Agent 4. `make_terminal_chooser()` is the terminal version: it shows
the top 5 (score, title, company, gaps) and accepts a number, `c` for a
custom role, or `q` / Ctrl-D to quit. Invalid input is explained and asked again.

**Fallbacks** (each one is printed as a `WARNING [Orchestrator]` and recorded):
if Agent 2 can't search (no `RAPIDAPI_KEY`, quota used up, a network or API
error, or no jobs found), the fake jobs in `tests/fixtures/jobs.json` are used.
Without a key, Agent 2 isn't called at all, so no Groq calls are wasted.
Agents 1 and 3 keep their own fallbacks (rule-based parser; TF-IDF and
rule-based key points).

**Stage trace** — `output/run_trace.json` (git-ignored), one entry per stage.
Agent 2's entry lists every real search (LLM steps, rule fallback, widenings)
and counts real JSearch jobs apart from fixture jobs:
```json
{"agent": "Agent 2: job search", "kind_of_ai": "generative LLM with tool use",
 "model": "Groq openai/gpt-oss-120b + JSearch (RapidAPI)", "seconds": 4.2,
 "fallback_used": false, "fallback": null,
 "details": {"jobs_source": "jsearch", "jobs_found": 10, "search_mode": "llm", "steps": 2,
             "real_jobs": 10, "fixture_jobs": 0, "widened_to": ["Louisiana"],
             "searches": [
   {"step": 1, "query_text": "Software Engineer Java OOP in Monroe, LA", "location": "Monroe, LA",
    "http_status": 200, "raw_count": 0, "returned": 0, "from_cache": false},
   {"step": "widen", "query_text": "Software Engineer in Louisiana", "location": "Louisiana",
    "http_status": 200, "raw_count": 10, "returned": 10, "from_cache": false,
    "widened_from": "Monroe, LA"}]}}
```
A failed search is printed as `WARNING [Orchestrator]: Agent 2 search failed
(HTTP 429) ...` and the fallback reason names the status, so "the source
failed" never reads as "found no jobs". The command line prints the searches
after the stage table.

**Agent 4 handoff** — `output/interview_brief.json` (git-ignored). This is
the contract with Agent 4 (`brief_version` changes if the format does):
```json
{
  "brief_version": 1,
  "created": "2026-11-16T18:30:00",
  "jobs_source": "jsearch",
  "job": {
    "id": "jsearch-abc123", "title": "Junior Java Developer", "company": "Maplewood Software",
    "location": "Monroe, LA", "score": 100.0,
    "description": "Full job posting text ...",
    "key_points": ["junior developer", "Java", "internal tools", "..."],
    "required_skills": ["Java", "Git", "OOP", "..."],
    "nice_to_have": ["C++"],
    "gaps": ["Linux"],
    "interview_focus": ["Linux (asked for, not on the resume)",
                        "Java (ask for an example; resume: skills: Java)", "..."]
  },
  "candidate": {
    "skills": ["Java", "C++", "Git", "..."],
    "projects": [{"name": "Car Rental System", "role": "Lead Developer", "date": "Jun 2024",
                  "summary": "...", "skills": ["Java", "Git", "OOP"]}],
    "experience_years": 2.6,
    "education": {"institution": "...", "degree": "Bachelor of Science in Computer Science",
                  "field": "Computer Science", "graduation": "May 2028"}
  }
}
```
- `job.interview_focus` is the list of topics for Agent 4 to ask about:
  gaps to probe first, then claims to check, each with its resume evidence.
- **No contact details:** the candidate part has no name, email, phone or
  links. Free text (the job description, project summaries, key points,
  gaps, interview topics) is also scrubbed of emails, phone numbers, links
  and the candidate's name ("the candidate").
- If the user quits, no brief is written.

---

## Tests

```bash
python -m pytest -v
```

The tests run offline with no API keys. GitHub Actions
(`.github/workflows/tests.yml`) runs them on every push and pull request;
see the **Actions** tab on GitHub, or the ✓ / ✗ next to each commit.

| File | Covers |
|---|---|
| `tests/test_llm.py` | `ask_llm`: Gemini/Groq requests, JSON + schema validation, missing keys, retries and `retry-after` |
| `tests/test_parser.py` | Agent 1 rule-based parser on 3 fake resumes with different heading, date and skill styles |
| `tests/test_parser_llm.py` | Agent 1 LLM parser: same format as rules, cleanup, every fallback case |
| `tests/test_ner_data.py` | The annotated NER data: offsets match the text, spans on token boundaries, no overlaps, every label present, labeling rules followed, file matches the generator |
| `tests/test_train_ner.py` | Train/test split, exact-match P/R/F1 on known cases, a 1-epoch training smoke run, the committed model's saved metrics |
| `tests/test_parser_ner.py` | NER parser with a fake "perfect" model: same profile as rules on the 3 fixtures, skill entity linking (aliases, unknown skills dropped, text matches added), planned-work skills dropped, company recovered from "Company | Title", Python date math, model mistakes filtered (fake dates, project roles, links), loud fallback when the model is missing, broken or crashes, pipeline default; plus the real committed model (F1 ≥ 0.90 on the hand-annotated resumes) |
| `tests/test_unseen_layouts.py` | The 3 fake resumes in unseen layouts are readable (the PDF really has two columns) and the checker's field comparison works |
| `tests/test_orchestrator.py` | The full run with fake Agent 2 LLM/provider: stage order and trace fields, fallbacks recorded, no contact details in the trace, fixture-job fallback (no key, failed searches, no jobs, crash) with a warning, per-search records and real vs fixture job counts in the trace, widening traced and warned, HTTP status in the error warning, no key = no Groq/JSearch calls, `--offline`; the terminal chooser (top 5 menu, number, invalid input asked again, custom role validated, quit / Ctrl-D); custom role through Agent 3; the interview brief (fields, original description, no contact details, free text scrubbed, none when the user quits) |
| `tests/test_job_synthesizer.py` | Requirement extraction, scoring, penalties, ranking (on the TF-IDF fallback), the optional LLM key points with a fake Groq: score unchanged, no contact details in the prompt, every fallback case |
| `tests/test_keypoints.py` | Embedding key points: spaCy candidate phrases, the real KeyBERT on a fake embedding model, resume matching (by skill, by similarity, not at all), nice-to-haves, field building, no Groq calls by default, score unchanged, fallbacks, the KeyBERT-vs-Groq matching in `scripts/compare_key_points.py` |
| `tests/test_embeddings.py` | Chunking, best-chunk similarity, the 0–40 mapping, scores not depending on other jobs, fallback |
| `tests/test_pipeline.py` | `run_pipeline` and the `/rank` endpoint (FastAPI `TestClient`) |
| `tests/test_job_search_agent.py` | Agent 2 loop with a scripted fake LLM: search/finish, merging, repeat queries, step and job limits, rule-based fallback, provider errors (status in the trace, logged warning), per-search records, widening city → state → Remote (and when not to widen), prompt contains no contact details, query cleaning |
| `tests/test_jsearch_provider.py` | JSearch provider with a fake HTTP call: request building, result parsing, error cases with their HTTP status, `last_request` (status, raw count, cache hits), key never in error messages; query cache in a temp folder: a hit skips the request, expired entries are refreshed, errors and empty results aren't cached, a corrupt file is ignored, `cache_path=None` turns it off |
| `tests/test_search_endpoint.py` | `POST /search`: NER parser used, success, fallback, empty results, bad input (file type, `limit`, `max_steps`), `max_steps` default and pass-through, 502 on job source failure |

**Tests never call a real LLM or job site, or load the real embedding model.**
`tests/conftest.py` blocks the network function in `agents/llm.py`, the
model loader in `agents/embeddings.py`, KeyBERT in `agents/keypoints.py` and
the NER loader in `agents/resume_parser.py` for every test. Tests that need
a reply, an embedding or entities install a fake one. The suite runs
offline, without API keys, and downloads nothing. The NER model is committed
in git, so the tests that load the real NER model always run, and they fail
if it's missing or broken.

All test data is fake: `tests/fixtures/` holds invented resumes (rebuild
them with `python tests/fixtures/make_fixtures.py`) and invented job postings.
The one test that uses `data/sample.pdf` is skipped when that file is missing.
Real resumes (`data/*.pdf`, `data/*.docx`) and `output/` are git-ignored.

---

## Known limitations

**Agent 1, NER path (default)**
- **Small, synthetic training data.** 48 generated resumes train the
  model. It works well on generated and simple hand-written resumes, but
  real resumes have more variety, and errors move around between retrains.
- **Evaluation is small and not fully blind.** The hand-written set has
  only 3 resumes (67 entities). It was also used to choose the training
  size (30 → 60 resumes), and the extra sections (coursework, languages,
  leadership) were added after looking at how the model failed on
  `data/sample.pdf`. Treat 0.942 as optimistic for real resumes.
- **Known model mistakes:** hyphenated names split (fixed by the name fallback when the name is the first line), hyphenated brand names
  ("Chick-fil-A") not seen as companies, place names or course names
  tagged as skills, a company and a job title swapped in unfamiliar layouts.
  Entity linking now drops skill entities that aren't known skills, the
  "Company | Title" split recovers a missed company or a company tagged as a
  title, and coursework skills come from the text match, so on sample.pdf and
  the 3 fixtures these no longer reach the profile (every job is right).
- **Only skills in `skills.txt` are kept.** Linking drops anything else,
  including real skills that aren't on the list ("Word", "Kubernetes").
  Adding a skill to `skills.txt` changes the NER training data generator,
  so the model must be retrained (`python scripts/train_ner.py`).
- **The future-work rule is word-based:** a real skill mentioned only in a
  sentence that also says "learning" or "plan" is dropped.
- **Unseen layouts.** `scripts/check_unseen_layouts.py` runs 3 fake resumes
  the model never saw (`tests/fixtures/unseen/`). NER gets **19/24** fields
  right, the rules 16/24 (before the season-date and name fixes: 16/24 and
  14/24). Skills and emails are right in all three. Still broken:
  - *Two-column PDF:* the PDF reader merges the columns line by line
    ("Python, SQL, Docker Harborline Logistics | …"), so a skill becomes part
    of a company name ("Docker Harborline Logistics"), the degree is cut at
    the column break ("Associate of Science in"), the graduation date is
    taken from the job next to it (Jan 2021 instead of May 2023), and the
    other column's heading sticks to a company ("EDUCATION Copperfield
    Diner"). (The name is now right.)
  - *No section headings:* the first job is lost because "Front-End
    Developer" was tagged ORG, not JOB_TITLE. The rule-based parser finds no
    jobs or education at all without headings. (The name is now right.)
  - *Season dates* (now read: Spring = Jan, Summer = Jun, Fall = Sep,
    Winter = Dec, so "Fall 2023 – Spring 2024" = 5 months): the second job
    is missing because "Software Lead" wasn't tagged as a job title. The
    rule-based parser gets this resume fully right.
- **Layout still matters** for grouping entities into jobs and education
  (same line or neighbouring lines).
- The trained model (15 MB) is committed in `models/resume_ner/`, so every
  clone uses the same model. Retraining is only needed after changing the
  training data; with the fixed seed it reproduces the same scores.

**Agent 1, LLM path (optional)**
- **Resume text is sent to Google (Gemini).** Real resumes are personal
  data, so the default NER parser (which runs locally) or `--rules` is safer.
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

**Agent 2, job search**
- **Quota:** the free JSearch plan has 200 requests a month, and each
  search costs one. One `/search` call uses 1 to `max_steps` (at most 4);
  searches already in the 24-hour cache cost nothing.
- **Cached results can be up to a day old.** Delete
  `output/jsearch_cache.json` for fresh postings.
- **Short queries only:** the query is the title plus 2 keywords, so a long
  skills list can't all be searched in one query. The agent can try different
  pairs across its steps, and Agent 3 ranks with the full resume.
- **Location:** the location you pass always wins, and the agent can't widen
  it. Small markets can return very few jobs (one test returned 1), so try a
  bigger city or leave it blank.
- **Rule-based fallback title:** it uses the first experience entry's title,
  which can be an odd title (for example "Student Webmaster") that returns
  nothing.
- **Search quality varies:** the LLM can pick poor queries, and output can
  differ between runs. Job titles and companies shown to the LLM are data
  from outside, so job postings could contain instructions aimed at the model.
- **Aggregated postings:** JSearch collects postings from other boards, so
  some may be stale or duplicated. Descriptions are full text, which Agent 3
  needs.
- **No experience-level filter:** the profile's level is not sent to JSearch.

**Agent 3, key points (KeyBERT + embeddings, default)**
- **No reasoning about the candidate.** It can't tell that a graduate
  doesn't fit a current-student internship. Degrees and fields are checked
  by exact rules, so "a related field" is accepted for any degree: rules
  can't judge whether two fields are related.
- **Phrases, not explanations.** Some picks are generic ("Bachelor",
  "Computer Science").
- **The kind rules are word rules.** A tool missing from `skills.txt`
  ("dashboards", "version control") counts as a role phrase, so it is never
  a gap.
- **The 0.45 / 0.55 thresholds** come from only 3 fake resumes and 6 jobs,
  and fit only this model.
- **Nice-to-haves are cue words**: only skills in a sentence with "plus",
  "bonus", "preferred", … are found.

**Agent 3, key points (optional LLM, off by default)**
- **Groq free-tier rate limit:** about 3 jobs per minute (88 s for 6 jobs
  in the comparison run). Jobs where Groq fails get the embedding version.
- **The LLM can misjudge the candidate**, its output varies between runs,
  and job postings could contain instructions aimed at the model.
- **Job text and a candidate summary (no contact details) are sent to Groq.**

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
