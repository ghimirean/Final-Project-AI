"""Agent 4: the mock interview (computer vision + speech).

Entry point: run_interview(brief_path). The orchestrator calls it after the
user picks a job, i.e. after it prints "Interview brief for Agent 4".

    python agents/interview_agent.py                      # output/interview_brief.json
    python agents/interview_agent.py --questions-only     # just print the questions

Steps (each added in its own commit):
  1. read the brief and build 4-5 questions from templates   (interview_questions.py)
  2. speech: record / read audio, Whisper, words per minute, fillers, pauses
  3. vision: record / read video, MediaPipe eye contact and posture
  4. write output/interview_results.json
"""
import argparse
import os
import sys

from interview_questions import BriefError, build_questions, load_brief

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "output")
DEFAULT_BRIEF = os.path.join(OUTPUT_DIR, "interview_brief.json")


def run_interview(brief_path=DEFAULT_BRIEF):
    """Reads the brief and builds the questions. Returns
    {"job": {...}, "questions": [...]}. Raises BriefError when there is no
    usable brief (so Agent 4 never starts from a missing or stale one)."""
    brief = load_brief(brief_path)
    job = brief["job"]
    return {"job": {"id": job.get("id"), "title": job.get("title"), "company": job.get("company")},
            "questions": build_questions(brief)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="INTERVAI Agent 4: mock interview")
    parser.add_argument("--brief", default=DEFAULT_BRIEF, help="interview brief (default output/interview_brief.json)")
    parser.add_argument("--questions-only", action="store_true", help="print the questions and stop")
    args = parser.parse_args(argv)
    try:
        interview = run_interview(args.brief)
    except BriefError as e:
        print(f"Agent 4 cannot start: {e}", file=sys.stderr)
        return 1
    job = interview["job"]
    print(f"Interview for {job['title']}" + (f" at {job['company']}" if job["company"] else ""))
    for q in interview["questions"]:
        print(f"\nQ{q['id']} [{q['kind']}] {q['question']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
