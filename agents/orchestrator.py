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
returns the chosen ranked job, or None to stop. A dashboard can pass its own.

If Agent 2 can't search (no JSearch key, quota used up, an error, or no jobs
found), the fake jobs in tests/fixtures/jobs.json are used instead, with a
visible WARNING. offline=True (--offline) uses them on purpose, to save the
JSearch quota during rehearsals.

Every stage is logged (agent, kind of AI, model, seconds, fallback) to
output/run_trace.json. The trace holds no name, email, phone or links.

Command line (from the project root):
    python agents/orchestrator.py data/sample.pdf --location "Monroe, LA"
    python agents/orchestrator.py data/sample.pdf --offline     # no JSearch / Groq calls
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

from embeddings import EMBED_CONFIG
from job_search_agent import LLM_CONFIG as AGENT2_LLM, MAX_STEPS, run_job_agent
from job_synthesizer import SKILLS_PATH, load_jobs, rank_jobs, save_json
from jsearch_provider import JSearchProvider
from resume_parser import NER_CONFIG, parse_resume_ner

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


def handoff_to_agent4(selected, profile, out_dir):
    """Hands the chosen job to Agent 4. (Step 1: records the choice only.)"""
    return {"job_id": selected["id"], "title": selected["title"]}


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

    # Human in the loop: the user picks one job
    t = time.time()
    selected = choose_job(ranked)
    trace.stage("hitl", time.time() - t,
                choice=(selected or {}).get("id"), title=(selected or {}).get("title"),
                quit=selected is None)

    # Agent 4 handoff
    handoff = None
    if selected is not None:
        t = time.time()
        handoff = handoff_to_agent4(selected, profile, out_dir)
        trace.stage("agent4", time.time() - t, **handoff)

    trace_path = trace.save(out_dir)
    public_profile = {k: v for k, v in profile.items() if k != "raw_text"}
    return {"profile": public_profile, "jobs_source": "fixture" if reason else "jsearch",
            "ranked_jobs": ranked, "selected": selected, "handoff": handoff,
            "trace_path": trace_path}


def choose_top_job(ranked):
    """Non-interactive choice: the best-scoring job."""
    return ranked[0] if ranked else None


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
    args = parser.parse_args()

    result = run_intervai(args.resume, args.location, choose_job=choose_top_job,
                          offline=args.offline, max_steps=args.max_steps)
    print(f"Jobs from: {result['jobs_source']}, ranked: {len(result['ranked_jobs'])}")
    if result["selected"]:
        print(f"Chosen: {result['selected']['title']} - {result['selected']['company']} "
              f"(score {result['selected']['score']})")
    print_trace(result["trace_path"])
    print(f"\nTrace saved to {os.path.relpath(result['trace_path'], ROOT)}")
