"""INTERVAI orchestrator: runs the four agents in order, in our own code
(no agent frameworks).

    Agent 1  resume parser       NER (fine-tuned spaCy)        -> profile
    Agent 2  job search          generative LLM + JSearch tool  -> job postings
    Agent 3  ranking/key points  semantic embeddings (MiniLM)   -> ranked jobs
    HITL     the user picks one job (choose_job callback)
    Agent 4  interview           computer vision + speech       <- handoff

    from orchestrator import run_intervai
    result = run_intervai("data/sample.pdf", "Monroe, LA", choose_job=my_callback)

choose_job(ranked_jobs) is called with Agent 3's ranking (best first) and
returns the chosen ranked job, {"custom_role": "Data Engineer"} to practise
for a role that isn't in the list, or None to stop. A dashboard can pass its
own; make_terminal_chooser() is the terminal version (top 5, pick a number,
'c' for a custom role, 'q' to quit).

If Agent 2 can't search (no JSearch key, quota used up, an error, or no jobs
found), the fake jobs in tests/fixtures/jobs.json are used instead, with a
visible WARNING. offline=True (--offline) uses them on purpose, to save the
JSearch quota during rehearsals.

demo=True (--demo) replays the most recent REAL search results without calling
Groq or JSearch, so rehearsals and the live demo show the same real jobs at no
cost. Every run where Agent 2 found real jobs saves them to
output/demo_jobs.json; --demo uses that file, or the newest entry of the
JSearch cache (output/jsearch_cache.json) if that is newer or the file is
missing. Agent 2's trace entry says "demo replay".

Every stage is logged (agent, kind of AI, model, seconds, fallback) to
output/run_trace.json. Agent 3's full ranking (best first, same format as
`python agents/job_synthesizer.py`) is saved to output/ranked_jobs.json. The chosen job and a candidate summary go to Agent 4
in output/interview_brief.json (format: README, "Orchestrator"). None of these files
holds the candidate's name, email, phone or links.

Command line (from the project root):
    python agents/orchestrator.py data/sample.pdf --location "Monroe, LA"
    python agents/orchestrator.py data/sample.pdf --offline     # no JSearch / Groq calls
    python agents/orchestrator.py data/sample.pdf --demo        # replay the last real jobs
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

from embeddings import EMBED_CONFIG
from job_search import JobDeduper, normalize_job
from job_search_agent import LLM_CONFIG as AGENT2_LLM, MAX_STEPS, run_job_agent
from job_synthesizer import SKILLS_PATH, load_jobs, rank_jobs, save_json
from jsearch_provider import DEFAULT_CACHE_PATH, JSearchProvider
from resume_parser import NER_CONFIG, URL_PATTERN, parse_resume_ner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "output")
FIXTURE_JOBS = os.path.join(ROOT, "tests", "fixtures", "jobs.json")
TOP_N = 5                      # jobs shown to the user to choose from

# What each stage is, for the trace (and the report)
STAGES = {
    "agent1": {"agent": "Agent 1: resume parser", "kind_of_ai": "NLP named-entity recognition",
               "model": f"spaCy en_core_web_sm fine-tuned ({os.path.relpath(NER_CONFIG['model_path'], ROOT)})"},
    "agent2": {"agent": "Agent 2: job search", "kind_of_ai": "generative LLM with tool use",
               "model": f"Groq {AGENT2_LLM['model']} + JSearch (RapidAPI)"},
    "agent3": {"agent": "Agent 3: ranking and key points", "kind_of_ai": "semantic embeddings",
               "model": f"{EMBED_CONFIG['model']} + KeyBERT"},
    "hitl": {"agent": "Human in the loop: job choice", "kind_of_ai": "human decision",
             "model": None},
    "agent4": {"agent": "Agent 4: interview (handoff)", "kind_of_ai": "computer vision + speech",
               "model": "MediaPipe + Whisper (Agent 4, separate module)"},
}


def warn(message):
    print(f"WARNING [Orchestrator]: {message}", file=sys.stderr)


class Trace:
    """Collects one entry per stage and writes them to output/run_trace.json."""

    def __init__(self):
        self.started = datetime.now().isoformat(timespec="seconds")
        self.stages = []

    def stage(self, key, seconds, fallback=None, **details):
        entry = dict(STAGES[key], seconds=round(seconds, 2),
                     fallback_used=fallback is not None, fallback=fallback, details=details)
        self.stages.append(entry)
        return entry

    def save(self, out_dir):
        path = os.path.join(out_dir, "run_trace.json")
        save_json({"started": self.started, "stages": self.stages}, path)
        return path


def search_records(trace):
    """One line per real search Agent 2 made (LLM steps, rule fallback and
    widenings): what was asked, where, the HTTP status and how many jobs came back."""
    records = []
    for step in trace:
        if "search" not in step:
            continue
        record = {"step": step["step"], **step["search"]}
        if step["step"] == "widen":
            record["widened_from"] = step.get("widened_from")
        records.append(record)
    return records


def warn_search_errors(records):
    """A visible WARNING for every failed search, with its HTTP status."""
    for r in records:
        if "error" in r:
            status = f"HTTP {r['http_status']}" if r.get("http_status") else "no HTTP reply"
            warn(f"Agent 2 search failed ({status}) for {r.get('query_text')!r}: {r['error']}")


# ---------- --demo: replay the last real search results ----------

DEMO_REPLAY = "demo replay"


def save_demo_jobs(jobs, location, details, out_dir):
    """After a real search: keeps its jobs in output/demo_jobs.json for --demo.
    Unlike the JSearch cache, this file never expires."""
    path = os.path.join(out_dir, "demo_jobs.json")
    save_json({"saved": datetime.now().isoformat(timespec="seconds"), "time": time.time(),
               "location": location or None, "searches": details.get("searches", []),
               "jobs": jobs}, path)
    return path


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def load_demo_jobs(out_dir, cache_path):
    """The most recent real jobs: output/demo_jobs.json or the newest JSearch
    cache entry (its 24-hour limit is ignored here), whichever is newer.
    Returns (jobs, info) or (None, None) if there are no real jobs saved."""
    options = []
    snapshot = _read_json(os.path.join(out_dir, "demo_jobs.json"))
    if isinstance(snapshot, dict) and snapshot.get("jobs"):
        options.append((snapshot.get("time") or 0, snapshot["jobs"], {
            "replayed_from": "output/demo_jobs.json (last real run)",
            "original_location": snapshot.get("location"),
            "searches": snapshot.get("searches", [])}))
    cache = _read_json(cache_path) if cache_path else None
    if isinstance(cache, dict):
        entries = [(v["time"], q, v["jobs"]) for q, v in cache.items()
                   if isinstance(v, dict) and isinstance(v.get("time"), (int, float))
                   and isinstance(v.get("jobs"), list) and v["jobs"]]
        if entries:
            saved, query, raw = max(entries)
            jobs = [j for j in (normalize_job(r, JSearchProvider.name) for r in raw) if j]
            options.append((saved, jobs, {"replayed_from": "output/jsearch_cache.json",
                                          "original_query": query, "searches": []}))
    if not options:
        return None, None
    saved, jobs, info = max(options, key=lambda o: o[0])
    seen = JobDeduper()
    jobs = [j for j in jobs if seen.add(j)]
    info["saved"] = datetime.fromtimestamp(saved).isoformat(timespec="seconds") if saved else None
    return (jobs or None), info


def search_jobs_with_fallback(profile, location, provider, offline, limit, max_steps,
                              demo=False, out_dir=OUTPUT_DIR):
    """Agent 2, or the fixture jobs.
    Returns (jobs, source "jsearch" | "fixture", fallback in words or None, details).
    details has one record per real search ("searches") and keeps real JSearch
    jobs ("real_jobs") apart from fixture jobs ("fixture_jobs")."""
    if offline:
        jobs = load_jobs(FIXTURE_JOBS)
        return (jobs, "fixture", "offline mode: fixture jobs used on purpose",
                {"real_jobs": 0, "fixture_jobs": len(jobs), "searches": []})
    if demo:
        jobs, info = load_demo_jobs(out_dir, getattr(provider, "cache_path", DEFAULT_CACHE_PATH))
        if jobs:
            jobs = jobs[:limit]
            reason = (f"{DEMO_REPLAY}: {len(jobs)} real JSearch jobs saved {info['saved']}, "
                      "no Groq or JSearch calls")
            print(f"NOTE [Orchestrator]: {reason} (from {info['replayed_from']})", file=sys.stderr)
            details = {"search_mode": DEMO_REPLAY, "real_jobs": len(jobs), "fixture_jobs": 0,
                       **info}
            return jobs, DEMO_REPLAY, reason, details
        reason = ("demo replay found no saved real jobs (run once without --demo first; "
                  "no Groq or JSearch calls were made)")
        warn(f"{reason}. Using the fake jobs in tests/fixtures/jobs.json instead.")
        jobs = load_jobs(FIXTURE_JOBS)
        return (jobs, "fixture", reason,
                {"search_mode": DEMO_REPLAY, "real_jobs": 0, "fixture_jobs": len(jobs),
                 "searches": []})
    details = {"real_jobs": 0, "searches": []}
    if not getattr(provider, "api_key", True):           # checked first: saves the Groq calls
        reason = "no JSearch key (RAPIDAPI_KEY not set in .env)"
    else:
        try:
            result = run_job_agent(profile, provider, location=location, limit=limit,
                                   max_steps=max_steps)
        except Exception as e:                           # quota used up, network, ...
            status = getattr(e, "status", None)
            reason = f"Agent 2 failed{f' (HTTP {status})' if status else ''}: {e}"
        else:
            records = search_records(result["trace"])
            warn_search_errors(records)
            details = {"search_mode": result["mode"], "steps": len(result["trace"]),
                       "real_jobs": len(result["jobs"]), "searches": records}
            widened = [r["location"] for r in records if r["step"] == "widen"]
            if widened:
                details["widened_to"] = widened
                warn(f"Agent 2 found no jobs in {location!r}, widened to: {', '.join(widened)}")
            llm_errors = [s["error"] for s in result["trace"] if "error" in s]
            if llm_errors:
                details["llm_error"] = llm_errors[0][:200]
            if result["jobs"]:
                if result["mode"] == "rules":            # real jobs, but the LLM didn't run
                    fallback = ("Agent 2's LLM failed, so its rule-based search query was used"
                                + (f" ({llm_errors[0][:120]})" if llm_errors else ""))
                    warn(fallback)
                    details["fixture_jobs"] = 0
                    return result["jobs"], "jsearch", fallback, details
                details["fixture_jobs"] = 0
                return result["jobs"], "jsearch", None, details
            errors = [f"HTTP {r['http_status']}: {r['error']}" if r.get("http_status")
                      else r["error"] for r in records if "error" in r]
            reason = (f"Agent 2's searches failed ({'; '.join(errors)})" if errors
                      else f"Agent 2 found no jobs ({len(records)} searches, "
                           "none failed, all returned 0 jobs)")
    warn(f"{reason}. Using the fake jobs in tests/fixtures/jobs.json instead.")
    jobs = load_jobs(FIXTURE_JOBS)
    details["fixture_jobs"] = len(jobs)
    return jobs, "fixture", reason, details


def agent3_fallbacks(ranked):
    """Which Agent 3 fallbacks were used, in words (None if none)."""
    used = []
    if any(r["breakdown"]["text_method"] != "embeddings" for r in ranked):
        used.append("TF-IDF instead of embeddings for the score")
    if any(r["assessed_by"] == "rules" for r in ranked):
        used.append("rule-based key points instead of KeyBERT")
    return "; ".join(used) or None


# ---------- Agent 4 handoff: output/interview_brief.json ----------

BRIEF_VERSION = 1
EMAIL = r"[\w.+-]+@[\w-]+\.[\w.-]+"
PHONE = r"\+?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"


def scrub(text, name=None):
    """Removes contact details from free text: emails, phone numbers, links,
    and the candidate's name (-> "the candidate")."""
    if not text:
        return text
    text = re.sub(EMAIL, "", text)
    text = re.sub(PHONE, "", text)
    text = re.sub(URL_PATTERN, "", text, flags=re.IGNORECASE)
    if name:
        text = re.sub(re.escape(name), "the candidate", text, flags=re.IGNORECASE)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def build_brief(selected, description, profile, jobs_source):
    """What Agent 4 needs to run the mock interview. The candidate part has
    skills, projects, experience_years and education ONLY: no name, email,
    phone or links (free text is scrubbed as well)."""
    a, name = selected["assessment"], profile.get("name")
    return {
        "brief_version": BRIEF_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "jobs_source": jobs_source,
        "job": {
            "id": selected["id"],
            "title": selected["title"],
            "company": selected.get("company"),
            "location": selected.get("location"),
            "score": selected.get("score"),
            "description": scrub(description, name),
            # interview_focus quotes resume lines as evidence, so lists are scrubbed too
            "key_points": [scrub(x, name) for x in a["key_points"]],
            "required_skills": a["required_skills"],
            "nice_to_have": a["nice_to_have"],
            "gaps": [scrub(x, name) for x in a["gaps"]],
            "interview_focus": [scrub(x, name) for x in a["interview_focus"]],
        },
        "candidate": {
            "skills": list(profile.get("skills", [])),
            "projects": [{"name": scrub(p.get("name"), name), "role": p.get("role"),
                          "date": p.get("date"), "summary": scrub(p.get("summary"), name),
                          "skills": list(p.get("skills", []))}
                         for p in profile.get("projects", [])],
            "experience_years": profile.get("experience_years"),
            "education": {k: (profile.get("education") or {}).get(k)
                          for k in ("institution", "degree", "field", "graduation")},
        },
    }


def save_ranked_jobs(ranked, profile, out_dir):
    """Writes Agent 3's full ranking to output/ranked_jobs.json and returns
    where. Text quoted from the resume is scrubbed like the interview brief."""
    name = profile.get("name")
    public = []
    for r in ranked:
        a = dict(r["assessment"])
        for key in ("key_points", "gaps", "interview_focus"):
            a[key] = [scrub(x, name) for x in a.get(key, [])]
        a["fit_summary"] = scrub(a.get("fit_summary"), name)
        public.append(dict(r, assessment=a))
    path = os.path.join(out_dir, "ranked_jobs.json")
    save_json(public, path)
    return path


def handoff_to_agent4(selected, description, profile, jobs_source, out_dir):
    """Writes output/interview_brief.json for Agent 4 and returns where."""
    path = os.path.join(out_dir, "interview_brief.json")
    save_json(build_brief(selected, description, profile, jobs_source), path)
    return {"brief": path, "job_id": selected["id"], "title": selected["title"]}


def run_intervai(resume_path, location="", choose_job=None, offline=False, provider=None,
                 limit=20, max_steps=MAX_STEPS, out_dir=OUTPUT_DIR, demo=False):
    """Runs Agent 1 -> Agent 2 -> Agent 3 -> HITL -> Agent 4 handoff.
    Returns {"profile", "jobs_source", "ranked_jobs", "selected", "handoff",
    "trace_path", "ranked_path"}. selected and handoff are None if the user quits.

    The same run in two halves, for a UI that can't wait inside a callback
    (a web dashboard): state = prepare_run(...), show state["ranked_jobs"],
    then finish_run(state, choice) once the user has chosen."""
    if choose_job is None:
        raise ValueError("choose_job is required: a function that picks one ranked job")
    state = prepare_run(resume_path, location, offline=offline, provider=provider,
                        limit=limit, max_steps=max_steps, out_dir=out_dir, demo=demo)
    return finish_run(state, choose_job(state["ranked_jobs"]))


def prepare_run(resume_path, location="", offline=False, provider=None,
                limit=20, max_steps=MAX_STEPS, out_dir=OUTPUT_DIR, demo=False):
    """First half of run_intervai: Agent 1 -> Agent 2 -> Agent 3, up to the
    job choice. Writes output/ranked_jobs.json. Returns the run state for
    finish_run(); state["ranked_jobs"] is what the user chooses from.
    The state holds the resume text (needed to score a custom role), so keep
    it in memory only and drop it after finish_run."""
    if offline and demo:
        raise ValueError("use offline (fake jobs) or demo (replay real jobs), not both")
    trace = Trace()

    # Agent 1: NER resume parser
    t = time.time()
    profile = parse_resume_ner(resume_path, skills_path=SKILLS_PATH)
    trace.stage("agent1", time.time() - t,
                fallback=None if profile["parsed_by"] == "ner" else "rule-based parser",
                parsed_by=profile["parsed_by"], skills=len(profile["skills"]),
                jobs=len(profile["experience"]))

    # Agent 2: LLM job search with the JSearch tool (or the fixture jobs)
    t = time.time()
    provider = provider or JSearchProvider()
    jobs, source, fallback, details = search_jobs_with_fallback(
        profile, location, provider, offline, limit, max_steps, demo=demo, out_dir=out_dir)
    if source == "jsearch":                              # real jobs: keep them for --demo
        save_demo_jobs(jobs, location, details, out_dir)
    trace.stage("agent2", time.time() - t, fallback=fallback,
                jobs_source=source, jobs_found=len(jobs),
                location=location or None, **details)

    # Agent 3: embeddings ranking + key points
    t = time.time()
    ranked = rank_jobs(profile, jobs)
    ranked_path = save_ranked_jobs(ranked, profile, out_dir)
    trace.stage("agent3", time.time() - t, fallback=agent3_fallbacks(ranked),
                jobs_ranked=len(ranked), top_score=ranked[0]["score"] if ranked else None,
                ranked_jobs=os.path.relpath(ranked_path, ROOT) if ranked_path.startswith(ROOT)
                else ranked_path)

    return {"profile": profile, "location": location, "jobs": jobs, "jobs_source": source,
            "ranked_jobs": ranked, "ranked_path": ranked_path, "out_dir": out_dir,
            "trace": trace, "hitl_started": time.time(), "finished": False}


def finish_run(state, selected):
    """Second half of run_intervai: records the user's choice (a ranked job,
    {"custom_role": "..."} or None to quit), writes the Agent 4 brief unless
    the user quit, and saves output/run_trace.json. Returns the same dict as
    run_intervai. A state can be finished only once."""
    if state.get("finished"):
        raise ValueError("this run is already finished; call prepare_run again")
    state["finished"] = True
    profile, location, jobs, source = (state["profile"], state["location"],
                                       state["jobs"], state["jobs_source"])
    ranked, trace, out_dir = state["ranked_jobs"], state["trace"], state["out_dir"]

    # Human in the loop: the user picks one job, types a custom role, or quits
    t = state["hitl_started"]
    custom = isinstance(selected, dict) and "custom_role" in selected
    if custom:
        selected = custom_role_job(selected["custom_role"], location, profile)
    trace.stage("hitl", time.time() - t,
                choice=(selected or {}).get("id"), title=(selected or {}).get("title"),
                custom_role=custom, quit=selected is None)

    # Agent 4 handoff: the interview brief
    handoff = None
    if selected is not None:
        t = time.time()
        descriptions = {job["id"]: job["description"] for job in jobs}
        description = descriptions.get(selected["id"], f"{selected['title']}.")
        handoff = handoff_to_agent4(selected, description, profile,
                                    source, out_dir)
        trace.stage("agent4", time.time() - t, job_id=handoff["job_id"], title=handoff["title"],
                    brief=os.path.relpath(handoff["brief"], ROOT) if handoff["brief"].startswith(ROOT)
                    else handoff["brief"])

    trace_path = trace.save(out_dir)
    public_profile = {k: v for k, v in profile.items() if k != "raw_text"}
    return {"profile": public_profile, "jobs_source": source,
            "ranked_jobs": ranked, "selected": selected, "handoff": handoff,
            "trace_path": trace_path, "ranked_path": state["ranked_path"]}


def choose_top_job(ranked):
    """Non-interactive choice: the best-scoring job."""
    return ranked[0] if ranked else None


# ---------- HITL: the terminal chooser ----------

MAX_ROLE_LENGTH = 80


def format_choices(ranked, top_n=TOP_N):
    """The menu: the top jobs with score, title, company and the first gaps."""
    lines = [f"\nTop {min(top_n, len(ranked))} matches (best first):"]
    for i, r in enumerate(ranked[:top_n], 1):
        gaps = r["assessment"]["gaps"]
        shown = ", ".join(gaps[:3]) + (f" (+{len(gaps) - 3} more)" if len(gaps) > 3 else "")
        lines.append(f"  {i}. [{r['score']:5.1f}] {r['title']} - {r['company'] or 'unknown company'}")
        lines.append(f"       gaps: {shown or 'none'}")
    lines.append(f"Pick a job (1-{min(top_n, len(ranked))}), 'c' for a custom role, or 'q' to quit.")
    return "\n".join(lines)


def make_terminal_chooser(ask=input, show=print, top_n=TOP_N):
    """Returns a choose_job callback that asks in the terminal. ask/show can
    be replaced (tests, or another text interface). The callback returns the
    chosen ranked job, {"custom_role": "..."} or None (quit)."""
    def choose(ranked):
        if not ranked:
            show("No jobs to choose from.")
            return None
        count = min(top_n, len(ranked))
        show(format_choices(ranked, top_n))
        try:
            while True:
                answer = ask("> ").strip().lower()
                if answer in ("q", "quit", "exit"):
                    return None
                if answer in ("c", "custom"):
                    role = ask("Role to practise for (e.g. Data Engineer): ").strip()
                    if not role:
                        show("The role can't be empty.")
                    elif len(role) > MAX_ROLE_LENGTH or not any(ch.isalpha() for ch in role):
                        show(f"Please type a job title (letters, at most {MAX_ROLE_LENGTH} characters).")
                    else:
                        return {"custom_role": role}
                    continue
                if answer.isdigit() and 1 <= int(answer) <= count:
                    return ranked[int(answer) - 1]
                show(f"Please type a number from 1 to {count}, 'c' or 'q'.")
        except (EOFError, KeyboardInterrupt):      # Ctrl-D / Ctrl-C: quit cleanly
            show("")
            return None
    return choose


def custom_role_job(role, location, profile):
    """A job for a role the user typed: Agent 3 ranks it like any posting, so
    Agent 4 gets the same fields. With no real posting, the role name is the
    description, so key points and gaps are thin."""
    job = {"id": "custom-role", "title": role, "company": None, "location": location or "",
           "url": None, "description": f"{role}."}
    return rank_jobs(profile, [job])[0]


def print_trace(path):
    with open(path, encoding="utf-8") as f:
        stages = json.load(f)["stages"]
    print(f"\n{'Stage':<34}{'Kind of AI':<32}{'Time':>7}  Fallback")
    for s in stages:
        print(f"{s['agent']:<34}{s['kind_of_ai']:<32}{s['seconds']:>6.1f}s  {s['fallback'] or '-'}")
    for s in stages:
        searches = s["details"].get("searches")
        if searches:
            replay = s["details"].get("search_mode") == DEMO_REPLAY
            print(f"\nAgent 2 searches{' (REPLAYED from the saved run, not made now)' if replay else ''}"
                  f" (real jobs: {s['details'].get('real_jobs', 0)}, "
                  f"fixture jobs: {s['details'].get('fixture_jobs', 0)}):")
            for r in searches:
                status = "cache" if r.get("from_cache") else (r.get("http_status") or "-")
                print(f"  {str(r['step']):<9}{str(status):<7}{str(r.get('raw_count')):>4} raw  "
                      f"{r.get('returned', 0):>3} kept  {r.get('query_text')!r}"
                      + (f"  ERROR: {r['error']}" if "error" in r else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the INTERVAI agents end to end.")
    parser.add_argument("resume", nargs="?", default=os.path.join(ROOT, "data", "sample.pdf"))
    parser.add_argument("--location", default="", help='e.g. "Monroe, LA" or "Remote"')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true",
                      help="use the fake fixture jobs instead of JSearch (saves the quota)")
    mode.add_argument("--demo", action="store_true",
                      help="replay the last real search results (no Groq or JSearch calls)")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS,
                        help=f"Agent 2 search steps, each at most one JSearch request (default {MAX_STEPS})")
    parser.add_argument("--auto", action="store_true",
                        help="pick the top job without asking (for scripted runs)")
    args = parser.parse_args()

    chooser = choose_top_job if args.auto else make_terminal_chooser()
    result = run_intervai(args.resume, args.location, choose_job=chooser,
                          offline=args.offline, max_steps=args.max_steps, demo=args.demo)
    print(f"Jobs from: {result['jobs_source']}, ranked: {len(result['ranked_jobs'])}")
    if result["selected"]:
        print(f"Chosen: {result['selected']['title']} - {result['selected']['company']} "
              f"(score {result['selected']['score']})")
        print(f"Interview brief for Agent 4: {os.path.relpath(result['handoff']['brief'], ROOT)}")
    else:
        print("No job chosen: stopped before Agent 4.")
    print_trace(result["trace_path"])
    print(f"\nTrace saved to {os.path.relpath(result['trace_path'], ROOT)}")
    print(f"Ranked jobs saved to {os.path.relpath(result['ranked_path'], ROOT)}")
