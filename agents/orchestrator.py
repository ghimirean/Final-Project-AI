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

Every stage is logged (agent, kind of AI, model, seconds, fallback) to
output/run_trace.json. The chosen job and a candidate summary go to Agent 4
in output/interview_brief.json (format: README, "Orchestrator"). Neither file
holds the candidate's name, email, phone or links.

Command line (from the project root):
    python agents/orchestrator.py data/sample.pdf --location "Monroe, LA"
    python agents/orchestrator.py data/sample.pdf --offline     # no JSearch / Groq calls
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

from embeddings import EMBED_CONFIG
from job_search_agent import LLM_CONFIG as AGENT2_LLM, MAX_STEPS, run_job_agent
from job_synthesizer import SKILLS_PATH, load_jobs, rank_jobs, save_json
from jsearch_provider import JSearchProvider
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


def search_jobs_with_fallback(profile, location, provider, offline, limit, max_steps):
    """Agent 2, or the fixture jobs. Returns (jobs, fallback reason or None, details)."""
    if offline:
        return load_jobs(FIXTURE_JOBS), "offline mode: fixture jobs used on purpose", {}
    if not getattr(provider, "api_key", True):           # checked first: saves the Groq calls
        reason = "no JSearch key (RAPIDAPI_KEY not set in .env)"
    else:
        try:
            result = run_job_agent(profile, provider, location=location, limit=limit,
                                   max_steps=max_steps)
        except Exception as e:                           # quota used up, network, ...
            reason = f"Agent 2 failed: {e}"
        else:
            details = {"search_mode": result["mode"], "steps": len(result["trace"])}
            if result["jobs"]:
                return result["jobs"], None, details
            errors = [s["search_error"] for s in result["trace"] if "search_error" in s]
            reason = (f"Agent 2's searches failed ({', '.join(errors)})" if errors
                      else "Agent 2 found no jobs")
    warn(f"{reason}. Using the fake jobs in tests/fixtures/jobs.json instead.")
    return load_jobs(FIXTURE_JOBS), reason, {}


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


def handoff_to_agent4(selected, description, profile, jobs_source, out_dir):
    """Writes output/interview_brief.json for Agent 4 and returns where."""
    path = os.path.join(out_dir, "interview_brief.json")
    save_json(build_brief(selected, description, profile, jobs_source), path)
    return {"brief": path, "job_id": selected["id"], "title": selected["title"]}


def run_intervai(resume_path, location="", choose_job=None, offline=False, provider=None,
                 limit=20, max_steps=MAX_STEPS, out_dir=OUTPUT_DIR):
    """Runs Agent 1 -> Agent 2 -> Agent 3 -> HITL -> Agent 4 handoff.
    Returns {"profile", "jobs_source", "ranked_jobs", "selected", "handoff",
    "trace_path"}. selected and handoff are None if the user quits."""
    if choose_job is None:
        raise ValueError("choose_job is required: a function that picks one ranked job")
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
    jobs, reason, details = search_jobs_with_fallback(profile, location, provider, offline,
                                                      limit, max_steps)
    trace.stage("agent2", time.time() - t, fallback=reason,
                jobs_source="fixture" if reason else "jsearch", jobs_found=len(jobs),
                location=location or None, **details)

    # Agent 3: embeddings ranking + key points
    t = time.time()
    ranked = rank_jobs(profile, jobs)
    trace.stage("agent3", time.time() - t, fallback=agent3_fallbacks(ranked),
                jobs_ranked=len(ranked), top_score=ranked[0]["score"] if ranked else None)

    # Human in the loop: the user picks one job, types a custom role, or quits
    t = time.time()
    selected = choose_job(ranked)
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
                                    "fixture" if reason else "jsearch", out_dir)
        trace.stage("agent4", time.time() - t, job_id=handoff["job_id"], title=handoff["title"],
                    brief=os.path.relpath(handoff["brief"], ROOT) if handoff["brief"].startswith(ROOT)
                    else handoff["brief"])

    trace_path = trace.save(out_dir)
    public_profile = {k: v for k, v in profile.items() if k != "raw_text"}
    return {"profile": public_profile, "jobs_source": "fixture" if reason else "jsearch",
            "ranked_jobs": ranked, "selected": selected, "handoff": handoff,
            "trace_path": trace_path}


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the INTERVAI agents end to end.")
    parser.add_argument("resume", nargs="?", default=os.path.join(ROOT, "data", "sample.pdf"))
    parser.add_argument("--location", default="", help='e.g. "Monroe, LA" or "Remote"')
    parser.add_argument("--offline", action="store_true",
                        help="use the fake fixture jobs instead of JSearch (saves the quota)")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS,
                        help=f"Agent 2 search steps, each at most one JSearch request (default {MAX_STEPS})")
    parser.add_argument("--auto", action="store_true",
                        help="pick the top job without asking (for scripted runs)")
    args = parser.parse_args()

    chooser = choose_top_job if args.auto else make_terminal_chooser()
    result = run_intervai(args.resume, args.location, choose_job=chooser,
                          offline=args.offline, max_steps=args.max_steps)
    print(f"Jobs from: {result['jobs_source']}, ranked: {len(result['ranked_jobs'])}")
    if result["selected"]:
        print(f"Chosen: {result['selected']['title']} - {result['selected']['company']} "
              f"(score {result['selected']['score']})")
        print(f"Interview brief for Agent 4: {os.path.relpath(result['handoff']['brief'], ROOT)}")
    else:
        print("No job chosen: stopped before Agent 4.")
    print_trace(result["trace_path"])
    print(f"\nTrace saved to {os.path.relpath(result['trace_path'], ROOT)}")
