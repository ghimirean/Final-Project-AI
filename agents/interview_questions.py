"""Agent 4, step 1: read the interview brief and build the interview questions.

The brief is output/interview_brief.json, written by the orchestrator after the
user picks a job (format: README, "Orchestrator"). The questions come from
fixed TEMPLATES filled with the brief's interview_focus and gaps. There is no
LLM here on purpose: Agent 4 is the computer vision + speech agent, and Agent 2
is the only LLM agent.

A run has 4 or 5 questions:
    1 intro  +  2-3 questions about the job (gaps first, then claims on the
    resume, then a project)  +  1 closing
"""
import json
import os
import re
from datetime import datetime

SUPPORTED_BRIEF_VERSIONS = (1,)
MAX_CONTENT_QUESTIONS = 3     # questions between the intro and the closing
MIN_CONTENT_QUESTIONS = 2     # filled with generic questions if the brief has too little
MAX_TOPIC_CHARS = 80

# interview_focus items from Agent 3 look like:
#   "Linux (asked for, not on the resume)"
#   "Java (ask for an example; resume: skills: Java)"
#   "Linux"                                  (rule-based fallback: a bare skill name)
#   "General fit for the Data Analyst role"  (placeholder when Agent 3 found nothing)
FOCUS_PATTERN = re.compile(r"^(?P<topic>.+?)\s+\((?P<note>asked for, not on the resume|ask for an example.*)\)\s*$",
                           re.DOTALL)
GENERAL_FIT = re.compile(r"^general fit for the .* role$", re.IGNORECASE)
YEARS_GAP = re.compile(r"(?P<needed>\d+)\+?\s+years of experience(?:\s*\(candidate has (?P<has>[\d.]+)\))?",
                       re.IGNORECASE)

GENERIC_QUESTIONS = [
    ("Describe a technical problem you found hard and how you worked through it.", "problem solving"),
    ("Tell me about a time you worked in a team. What was your part, and what would you do differently?",
     "teamwork"),
    ("Tell me about a time you had to learn something new quickly. How did you go about it?", "learning"),
]


class BriefError(Exception):
    """The interview brief is missing, unreadable, from an old run, or in a format Agent 4 doesn't know."""


# ---------- reading the brief ----------

def load_brief(path):
    """Reads and checks the interview brief. Raises BriefError with a message
    the user can act on."""
    if not os.path.isfile(path):
        raise BriefError(f"No interview brief at {path}. Run the orchestrator, choose a job, and wait for "
                         "'Interview brief for Agent 4' before starting the interview.")
    try:
        with open(path, encoding="utf-8") as f:
            brief = json.load(f)
    except (OSError, ValueError) as e:
        raise BriefError(f"Could not read the interview brief {path}: {e}")
    if not isinstance(brief, dict) or not isinstance(brief.get("job"), dict):
        raise BriefError(f"{path} is not an interview brief (no 'job' section).")
    if brief.get("brief_version") not in SUPPORTED_BRIEF_VERSIONS:
        raise BriefError(f"Interview brief version {brief.get('brief_version')!r} is not supported "
                         f"(Agent 4 reads version {SUPPORTED_BRIEF_VERSIONS[-1]}).")
    if not str(brief["job"].get("title") or "").strip():
        raise BriefError(f"{path} has no job title.")
    check_not_stale(brief, path)
    return brief


def check_not_stale(brief, path):
    """The orchestrator deletes the old brief when a run starts, so a brief
    that is older than the run_trace.json next to it is left over from an
    earlier run. Skipped when there is no trace or a time can't be read."""
    trace_path = os.path.join(os.path.dirname(os.path.abspath(path)), "run_trace.json")
    try:
        with open(trace_path, encoding="utf-8") as f:
            started = datetime.fromisoformat(json.load(f)["started"])
        created = datetime.fromisoformat(brief["created"])
    except (OSError, ValueError, KeyError, TypeError):
        return
    if created < started:
        raise BriefError("The interview brief is older than the latest orchestrator run, so it is from an "
                         "earlier run. Run the orchestrator again and choose a job.")


# ---------- building the questions ----------

def clean_topic(text):
    text = re.sub(r"\s+", " ", str(text or "")).strip(" .;,")
    return text[:MAX_TOPIC_CHARS].rstrip()


def parse_focus_item(item, gaps):
    """One interview_focus item -> {"kind", "topic", "detail"} or None.
    kind: "gap" (asked for, not on the resume), "claim" (on the resume, ask
    for an example), "experience" (years gap)."""
    item = str(item or "").strip()
    if not item or GENERAL_FIT.match(item):
        return None
    match = FOCUS_PATTERN.match(item)
    if match:
        topic, note = clean_topic(match["topic"]), match["note"]
        kind = "gap" if note.startswith("asked for") else "claim"
    else:
        topic = clean_topic(re.sub(r"\s*\(.*$", "", item, flags=re.DOTALL) or item)
        # a bare name (rule-based fallback): it is a gap if Agent 3 lists it as one
        kind = "gap" if topic.lower() in {clean_topic(g).lower() for g in gaps} else "claim"
    years = YEARS_GAP.search(item)
    if years:
        detail = years.groupdict()
        if not detail["has"]:                      # the number the candidate has is in the matching gaps entry
            for gap in gaps:
                found = YEARS_GAP.search(str(gap))
                if found and found["needed"] == detail["needed"] and found["has"]:
                    detail["has"] = found["has"]
                    break
        return {"kind": "experience", "topic": topic, "detail": detail}
    return {"kind": kind, "topic": topic, "detail": None} if topic else None


def parse_gap_item(item):
    """One gaps item (a skill, a phrase, "5+ years of experience (candidate has
    2.6)", "Bachelor's degree") -> the same shape as parse_focus_item."""
    item = str(item or "").strip()
    if not item:
        return None
    years = YEARS_GAP.search(item)
    if years:
        return {"kind": "experience", "topic": clean_topic(re.sub(r"\s*\(.*$", "", item)), "detail": years.groupdict()}
    topic = clean_topic(re.sub(r"\s*\(.*$", "", item, flags=re.DOTALL) or item)
    return {"kind": "gap", "topic": topic, "detail": None} if topic else None


def question_for(item):
    """Fills the template for one topic. Returns the question text."""
    topic, kind = item["topic"], item["kind"]
    if kind == "gap":
        return (f"The posting asks for {topic}, but I don't see it on your resume. What experience do you "
                "have with it, and how would you get up to speed quickly?")
    if kind == "claim":
        return (f"Your resume shows {topic}. Walk me through a specific example where you used it: "
                "the situation, what you did, and the result.")
    if kind == "experience":
        needed, has = (item["detail"] or {}).get("needed"), (item["detail"] or {}).get("has")
        have = f"about {has}" if has else "less"
        return (f"This role asks for {needed}+ years of experience, and you have {have}. "
                "Why are you ready for it anyway?")
    return f"The posting asks for {topic}. How does your background match that?"


def project_item(candidate):
    """The first project on the resume as a question topic, or None."""
    for project in (candidate or {}).get("projects") or []:
        name = clean_topic((project or {}).get("name"))
        if name:
            return {"kind": "project", "topic": name, "detail": (project or {}).get("role")}
    return None


def build_questions(brief):
    """Builds 4-5 questions from a (checked) brief. Each question is
    {"id", "kind", "topic", "source", "question"}. kind: intro, gap, claim,
    experience, project, general or closing."""
    job, candidate = brief["job"], brief.get("candidate") or {}
    title = clean_topic(job.get("title"))
    company = clean_topic(job.get("company"))
    gaps = job.get("gaps") or []

    content, seen = [], set()

    def add(item, source):
        if item and len(content) < MAX_CONTENT_QUESTIONS and item["topic"].lower() not in seen:
            seen.add(item["topic"].lower())
            content.append(dict(item, source=source))

    for focus in job.get("interview_focus") or []:
        add(parse_focus_item(focus, gaps), "interview_focus")
    for gap in gaps:
        add(parse_gap_item(gap), "gaps")
    add(project_item(candidate), "projects")

    for text, topic in GENERIC_QUESTIONS:
        if len(content) >= MIN_CONTENT_QUESTIONS:
            break
        content.append({"kind": "general", "topic": topic, "detail": None, "source": "template",
                        "text": text})

    at_company = f" at {company}" if company else ""
    questions = [{"kind": "intro", "topic": None, "source": "template",
                  "question": f"Tell me about yourself and why you are interested in the {title} role{at_company}."}]
    for item in content:
        if item["kind"] == "project":
            role = f" as {item['detail']}" if item["detail"] else ""
            text = (f"Tell me about your project {item['topic']}{role}. What was the hardest decision you made, "
                    "and what would you change?")
        else:
            text = item.get("text") or question_for(item)
        questions.append({"kind": item["kind"], "topic": item["topic"], "source": item["source"], "question": text})
    questions.append({"kind": "closing", "topic": None, "source": "template",
                      "question": f"Why should we hire you for the {title} role? Finish with the one point "
                                  "you most want us to remember."})
    for number, question in enumerate(questions, start=1):
        question["id"] = number
    return [{k: q[k] for k in ("id", "kind", "topic", "source", "question")} for q in questions]
