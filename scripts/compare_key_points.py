"""Compares Agent 3's key points: KeyBERT + embeddings (default) vs. Groq LLM.

Run from project root:
    python scripts/compare_key_points.py                         # fake fixture resume
    python scripts/compare_key_points.py path/to/resume.docx

Needs GROQ_API_KEY in .env for the LLM side (6 Groq calls; the free tier
allows about 3 per minute, so this takes ~2 minutes). Uses only fake data
by default: tests/fixtures/resume_numeric_dates.docx + tests/fixtures/jobs.json.

For each job prints both versions side by side, then a summary:
- time and API calls for each method
- "Groq points captured": for each Groq key point, the most similar
  KeyBERT phrase (embedding similarity); a point counts as captured at
  >= CAPTURED_AT. Groq writes sentences and KeyBERT picks phrases, so this
  measures whether the same topics were found, not the same words.
Results are also saved to output/key_points_comparison.json (git-ignored).
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from embeddings import embed  # noqa: E402
from job_synthesizer import LLM_CONFIG, SKILLS_PATH, load_jobs, rank_jobs, save_json  # noqa: E402
from resume_parser import parse_resume_ner  # noqa: E402

DEFAULT_RESUME = os.path.join(ROOT, "tests", "fixtures", "resume_numeric_dates.docx")
JOBS_PATH = os.path.join(ROOT, "tests", "fixtures", "jobs.json")
CAPTURED_AT = 0.5
WIDTH = 52


def side_by_side(left_title, left, right_title, right):
    print(f"   {left_title:<{WIDTH}} | {right_title}")
    for i in range(max(len(left), len(right))):
        a = left[i] if i < len(left) else ""
        b = right[i] if i < len(right) else ""
        print(f"   {a[:WIDTH]:<{WIDTH}} | {b}")


def captured(llm_points, keybert_points):
    """For each LLM point: (best KeyBERT phrase, similarity)."""
    if not llm_points or not keybert_points:
        return []
    sims = embed(llm_points) @ embed(keybert_points).T
    return [(keybert_points[int(row.argmax())], float(row.max())) for row in sims]


if __name__ == "__main__":
    resume = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESUME
    jobs = load_jobs(JOBS_PATH)
    profile = parse_resume_ner(resume, skills_path=SKILLS_PATH)
    print(f"resume: {os.path.basename(resume)} (parsed_by={profile['parsed_by']}), {len(jobs)} jobs")

    t = time.time()
    by_embeddings = {r["id"]: r for r in rank_jobs(profile, jobs)}
    embed_seconds = time.time() - t

    t = time.time()
    by_llm = {r["id"]: r for r in rank_jobs(profile, jobs, use_llm=True, top_n=len(jobs))}
    llm_seconds = time.time() - t

    report, scores = [], []
    for job in jobs:
        e, g = by_embeddings[job["id"]], by_llm[job["id"]]
        ea, ga = e["assessment"], g["assessment"]
        print(f"\n=== {job['id']} {job['title']} ({job['company']})   score {e['score']}"
              f"   [{e['assessed_by']} | {g['assessed_by']}]")
        side_by_side("KeyBERT + embeddings: key points", ea["key_points"],
                     f"Groq {LLM_CONFIG['model']}: key points", ga["key_points"])
        side_by_side("gaps", ea["gaps"], "gaps", ga["gaps"])
        side_by_side("interview focus", ea["interview_focus"], "interview focus", ga["interview_focus"])
        pairs = captured(ga["key_points"], ea["key_points"]) if g["assessed_by"] == "llm" else []
        for point, (phrase, sim) in zip(ga["key_points"], pairs):
            mark = "captured" if sim >= CAPTURED_AT else "missed"
            print(f"   {mark:>8} {sim:.2f}  {point[:60]}  <->  {phrase}")
        scores += [s for _, s in pairs]
        report.append({"id": job["id"], "title": job["title"], "score": e["score"],
                       "embeddings": ea, "llm": ga, "llm_assessed_by": g["assessed_by"],
                       "llm_point_best_match": [{"llm_point": p, "keybert_phrase": k, "similarity": round(s, 3)}
                                                for p, (k, s) in zip(ga["key_points"], pairs)]})

    n_llm = sum(1 for r in by_llm.values() if r["assessed_by"] == "llm")
    print("\nSUMMARY")
    print(f"  KeyBERT + embeddings: {embed_seconds:.1f} s for {len(jobs)} jobs, 0 API calls, runs offline")
    print(f"  Groq LLM:             {llm_seconds:.1f} s for {len(jobs)} jobs, {n_llm} answered by Groq")
    if scores:
        hits = sum(s >= CAPTURED_AT for s in scores)
        print(f"  Groq points captured by a KeyBERT phrase (similarity >= {CAPTURED_AT}): "
              f"{hits}/{len(scores)}; mean best similarity {sum(scores) / len(scores):.2f}")
    print("  Scores are identical (key points never change the score):",
          all(by_embeddings[i]["score"] == by_llm[i]["score"] for i in by_embeddings))
    save_json({"resume": os.path.basename(resume), "embed_seconds": embed_seconds,
               "llm_seconds": llm_seconds, "jobs": report},
              os.path.join(ROOT, "output", "key_points_comparison.json"))
    print("Saved output/key_points_comparison.json")
