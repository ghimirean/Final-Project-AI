# INTERVAI

A multi-agent AI mock interview platform. This README covers **Agent 1
(Resume Parser)**, **Agent 3 (Job Description Synthesizer + Match Scoring)**
and the **pipeline / API** that connects them to the HITL dashboard and Agent 4.

| Agent | Kind of AI | Model | What the model does | Fallback |
|---|---|---|---|---|
| Agent 1 | **NLP: named-entity recognition** (supervised, trained by us) | spaCy `en_core_web_sm`, fine-tuned on 48 annotated resumes → `models/resume_ner/` | Finds names, companies, schools, degrees, job titles, skills and dates in the resume | Rule-based parser |
| Agent 3 | **Large language model** | Groq `qwen/qwen3.8-27b` | Key points per job: what the employer wants, fit, gaps, interview topics | Rule-based key points |
| Agent 3 | **Semantic embeddings** | local `all-MiniLM-L6-v2` (sentence-transformers) | Text-similarity part of the match score | TF-IDF |

Agent 1 can also use Gemini (`gemini-3.5-flash`) as an alternative parser
(`parser="llm"`). It's kept for comparison and isn't the default.

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
  one-line change. Agent 3 uses `{"provider": "groq", "model": "qwen/qwen3.8-27b"}`.
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
   - name = first `PERSON`
   - skills = every `SKILL` outside a link, spelled like `skills.txt` when listed there; new skills are kept too
   - education = first `DEGREE`, the `ORG` up to 2 lines above or 1 below it, and the nearest date
   - jobs = one per line with a `JOB_TITLE`, with the `ORG` on the same or a neighbouring line and the next two real dates (job titles inside Projects, Leadership or Awards sections are ignored)
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
  python scripts/train_ner.py     # retrains and saves models/resume_ner/ (15 MB, git-ignored)
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
| `data/sample.pdf` | NER misses the company "Chick-fil-A", adds a place name and a course name as skills, and keeps "Microsoft Excel" whole; Gemini adds "Word"; jobs, months and education agree |

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

- **Only the top N jobs** (default 5, `LLM_CONFIG["top_n"]`) get LLM key
  points. Jobs are scored and sorted first, then the best N are sent to
  Groq. The rest get the rule-based version. This keeps long job lists fast
  and inside the rate limit.
- **Accurate candidate facts.** If the graduation date is in the future,
  the summary says the degree is `expected May 2028, in progress`, and the
  prompt tells the model it must not claim the degree is completed. The
  total years are labeled as "all jobs, related or not", and the model is
  told to count only experience relevant to the posting (a cashier job
  doesn't count toward "1+ years building websites").
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

run_pipeline("data/sample.pdf", jobs, parser="llm")    # Agent 1 uses Gemini instead
run_pipeline("data/sample.pdf", jobs, use_llm=False)   # no API calls at all
```
It checks the job list first and raises `ValueError` if a job is missing fields.
By default Agent 1 parses with the local NER model (`parser="ner"`) and Agent 3
writes key points with Groq (`use_llm=True`), each with its own fallback.
`parser` can be `"ner"`, `"llm"` or `"rules"`. `use_llm=False` makes no API
calls at all (an `"llm"` parser then becomes `"rules"`). The API always uses
the defaults.

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
| `tests/test_parser_llm.py` | Agent 1 LLM parser: same format as rules, cleanup, every fallback case |
| `tests/test_ner_data.py` | The annotated NER data: offsets match the text, spans on token boundaries, no overlaps, every label present, labeling rules followed, file matches the generator |
| `tests/test_train_ner.py` | Train/test split, exact-match P/R/F1 on known cases, a 1-epoch training smoke run, saved model metrics (if built) |
| `tests/test_parser_ner.py` | NER parser with a fake "perfect" model: same profile as rules on the 3 fixtures, Python date math, model mistakes filtered (fake dates, project roles, links), fallback when the model is missing or crashes, pipeline default; plus the real model (if built) |
| `tests/test_job_synthesizer.py` | Requirement extraction, scoring, penalties, ranking (on the TF-IDF fallback), LLM key points with a fake Groq: score unchanged, no contact details in the prompt, every fallback case |
| `tests/test_embeddings.py` | Chunking, best-chunk similarity, the 0–40 mapping, scores not depending on other jobs, fallback |
| `tests/test_pipeline.py` | `run_pipeline` and the `/rank` endpoint (FastAPI `TestClient`) |

**Tests never call a real LLM or load the real embedding or NER model.**
`tests/conftest.py` blocks the network function in `agents/llm.py`, the
model loader in `agents/embeddings.py` and the NER loader in
`agents/resume_parser.py` for every test. Tests that need a reply, an
embedding or entities install a fake one. The suite runs offline, without
API keys, downloads nothing, and passes without a trained model (the few
tests that use the real NER model are skipped when it isn't built).

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
- **Known mistakes:** hyphenated names split, hyphenated brand names
  ("Chick-fil-A") not seen as companies, place names or course names
  tagged as skills, a company and a job title swapped in unfamiliar layouts.
- **Course names aren't skills by our labeling rules,** so a skill that
  appears only in "Relevant Coursework" (e.g. "Algorithms") isn't counted.
  That lowers Agent 3 scores compared to the rules (90 vs 100 on sample.pdf).
- **Layout still matters** for grouping entities into jobs and education
  (same line or neighbouring lines).
- The model must be trained locally (`python scripts/train_ner.py`), since
  it's git-ignored. Until then Agent 1 uses the rules.

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

**Agent 3, key points (LLM)**
- **Groq free-tier rate limit:** `qwen/qwen3.8-27b` allows 1,000 output
  tokens per minute, about 3 jobs. With the default top 5, a run waits on
  retries (about 1.3 minutes), and if a wait would exceed the retry budget,
  that job falls back to rule-based key points. Check `assessed_by`.
- **The LLM can misjudge the candidate.** Before the fixes above it said a
  student still studying "possesses the required Bachelor's degree" and
  counted cashier years toward "1+ years building websites". Both are fixed
  in our test run, but the model can still misjudge relevance (e.g. whether
  a "Test Lead" job was technical). Key points are guidance for the
  interview agent, not facts. The score doesn't depend on them.
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
