"""INTERVAI HITL dashboard: the human-in-the-loop job choice, in the browser.

    streamlit run dashboard.py

The user uploads a resume, picks a location and a mode, and clicks
"Find jobs". The orchestrator's own functions do the work (no logic is
duplicated here):

    prepare_run()  Agent 1 -> Agent 2 -> Agent 3   (once, on "Find jobs")
    finish_run()   the user's choice -> Agent 4 brief + run trace

Streamlit reruns this whole script on every click, so prepare_run is called
ONLY when "Find jobs" is pressed; its result is kept in st.session_state.

Modes:
    demo     replays the last real search results (no Groq or JSearch calls)
    offline  the 6 fake jobs in tests/fixtures/jobs.json (no Groq or JSearch calls)
    live     real Groq + JSearch search; spends JSearch requests (free plan:
             about 200 a month), so it defaults to 1 search step
"""
import os
import re
import sys
import tempfile

import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "agents"))
from orchestrator import (DEMO_REPLAY, MAX_ROLE_LENGTH, MAX_STEPS,  # noqa: E402
                          OUTPUT_DIR, STAGES, finish_run, prepare_run)

ALLOWED_TYPES = ("pdf", "docx")
MODES = {
    "demo": "Demo: replay the last real jobs (no API calls)",
    "offline": "Offline: 6 fake practice jobs (no API calls)",
    "live": "Live: search JSearch now (spends requests)",
}
LIVE_DEFAULT_STEPS = 1
CUSTOM = "custom"          # the choice value for "a role that isn't in the list"


# ---------- non-UI logic (plain functions, tested without Streamlit) ----------

def output_dir():
    """Where the run files go: output/ (like the terminal version), or the
    folder in INTERVAI_OUTPUT_DIR (the tests use a temp folder)."""
    return os.environ.get("INTERVAI_OUTPUT_DIR") or OUTPUT_DIR


def mode_options(mode, max_steps=LIVE_DEFAULT_STEPS):
    """prepare_run keyword arguments for a dashboard mode."""
    if mode == "demo":
        return {"demo": True}
    if mode == "offline":
        return {"offline": True}
    if mode == "live":
        if not 1 <= int(max_steps) <= MAX_STEPS:
            raise ValueError(f"max_steps must be between 1 and {MAX_STEPS}")
        return {"max_steps": int(max_steps)}
    raise ValueError(f"unknown mode: {mode!r}")


def file_extension(filename):
    """'.pdf' or '.docx' (lower case), or raises ValueError for anything else."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext.lstrip(".") not in ALLOWED_TYPES:
        raise ValueError(f"the resume must be a .pdf or .docx file, got '{ext or 'no extension'}'")
    return ext


RUN_KEYS = ("run", "picked", "role")   # session keys that belong to one run


def clear_run_state(session):
    """Forgets everything about a finished run that could hold resume data:
    the run state (it holds the resume text), the choice widgets, and the
    uploaded file itself (a new uploader key drops Streamlit's copy)."""
    for key in RUN_KEYS:
        session.pop(key, None)
    session["upload_key"] = session.get("upload_key", 0) + 1


def run_agents(filename, data, location, mode, max_steps=LIVE_DEFAULT_STEPS,
               prepare=prepare_run, on_stage=None):
    """Writes the uploaded resume to a temp file, runs Agents 1-3 on it and
    deletes the file again, even if an agent fails. Returns the run state.
    on_stage("agent1" | "agent2" | "agent3") is called as each stage starts."""
    options = mode_options(mode, max_steps)
    ext = file_extension(filename)
    fd, path = tempfile.mkstemp(suffix=ext, prefix="intervai_resume_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return prepare(path, (location or "").strip(), out_dir=output_dir(),
                       on_stage=on_stage, **options)
    finally:
        os.remove(path)


# ---------- loading and error messages ----------

def stage_message(key, mode):
    """What the spinner says while a stage runs."""
    if key == "agent1":
        return "Agent 1: reading your resume (NER model)..."
    if key == "agent2":
        return {"live": "Agent 2: searching for jobs (Groq + JSearch, can take 10-30 seconds)...",
                "demo": "Agent 2: replaying the last real search results (no API calls)...",
                "offline": "Agent 2: loading the fake practice jobs (no API calls)..."}.get(
                    mode, "Agent 2: finding jobs...")
    if key == "agent3":
        return "Agent 3: ranking the jobs and writing key points (embeddings)..."
    return "Working..."


def failure_message(stage, error):
    """A clear message for a stage that raised an error."""
    what = f"{type(error).__name__}: {error}"
    if stage in STAGES:
        return f"{STAGES[stage]['agent']} failed, so no jobs can be shown. ({what})"
    return f"Could not start the run. ({what})"


def stage_entry(state, key):
    """The run-trace entry for one stage (agent1/agent2/agent3), or {}."""
    name = STAGES[key]["agent"]
    for entry in state["trace"].stages:
        if entry["agent"] == name:
            return entry
    return {}


def stage_notices(state, mode):
    """(level, message) pairs about fallbacks and empty results, in the order
    they should be shown. level is "error", "warning" or "info"."""
    notices = []
    agent1, agent2, agent3 = (stage_entry(state, k) for k in ("agent1", "agent2", "agent3"))
    if agent1.get("fallback_used"):
        notices.append(("info", "Agent 1 could not use the NER model, so the rule-based "
                                "parser read your resume."))
    source = state.get("jobs_source")
    reason = str(agent2.get("fallback") or "")
    details = agent2.get("details") or {}
    if source == DEMO_REPLAY:
        notices.append(("info", f"Demo mode: replaying {details.get('real_jobs', 0)} real jobs "
                                f"saved {details.get('saved') or 'earlier'} (no API calls)."))
    elif source == "fixture" and mode == "offline":
        notices.append(("info", "Offline mode: these are fake practice jobs (no API calls)."))
    elif source == "fixture" and mode == "demo":
        notices.append(("warning", "Demo mode found no saved real jobs, so these are fake "
                                   "practice jobs. Do one live run first to save real jobs."))
    elif source == "fixture":
        if "429" in reason or "quota" in reason.lower():
            notices.append(("error", "The job source is out of requests (JSearch quota used up "
                                     "or rate limit, HTTP 429). Showing fake practice jobs "
                                     "instead. Use demo mode, or wait for the quota to reset."))
        elif "no JSearch key" in reason:
            notices.append(("error", "No JSearch key: RAPIDAPI_KEY is not set in .env. "
                                     "Showing fake practice jobs instead."))
        elif "found no jobs" in reason:
            notices.append(("warning", "The live search found no jobs. Showing fake practice "
                                       "jobs instead. Try a bigger city, or leave the location "
                                       "blank."))
        else:
            notices.append(("error", f"The job search failed ({md(reason, 200)}). "
                                     "Showing fake practice jobs instead."))
    elif source == "jsearch" and agent2.get("fallback_used"):
        notices.append(("info", "Agent 2's LLM could not plan the search, so a rule-based "
                                "search query was used."))
    if details.get("widened_to"):
        notices.append(("info", "No jobs in that location, so the search was widened to: "
                                + ", ".join(md(x) for x in details["widened_to"]) + "."))
    if agent3.get("fallback_used"):
        notices.append(("info", f"Agent 3 used a fallback: {md(agent3.get('fallback'))}."))
    if not state.get("ranked_jobs"):
        notices.append(("warning", "No jobs were found to choose from. You can still type "
                                   "a role to practise for."))
    return notices


# Job titles, companies, descriptions and links come from the internet, so
# they are untrusted: no HTML is ever rendered (never unsafe_allow_html), and
# markdown characters are escaped so a posting can't add links or formatting.
MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:=&\"'^])")


def md(text, limit=300):
    """Untrusted text -> safe for st.markdown: one line, at most `limit`
    characters, every markdown character escaped."""
    text = " ".join(str(text if text is not None else "").split())
    if len(text) > limit:
        text = text[:limit - 1].rstrip() + "…"
    return MARKDOWN_SPECIAL.sub(r"\\\1", text)


def safe_url(url):
    """The link if it is a plain http:// or https:// URL, else None
    (javascript:, data:, relative links, ... are never made clickable)."""
    if not isinstance(url, str):
        return None
    url = url.strip()
    if not url.lower().startswith(("http://", "https://")) or len(url) > 2000:
        return None
    if any(ch.isspace() or ord(ch) < 32 for ch in url):
        return None
    return url


def bullets(items, empty="none"):
    """Untrusted list -> a safe markdown bullet list."""
    items = [md(x) for x in items or [] if str(x).strip()]
    return "\n".join(f"- {x}" for x in items) if items else f"_{empty}_"


def role_problem(role):
    """Why a typed custom role can't be used, or None if it's fine. Same
    rules as the terminal chooser."""
    role = (role or "").strip()
    if not role:
        return "Type the role you want to practise for."
    if len(role) > MAX_ROLE_LENGTH or not any(ch.isalpha() for ch in role):
        return f"Please type a job title (letters, at most {MAX_ROLE_LENGTH} characters)."
    return None


def make_choice(ranked, picked, role=""):
    """The value finish_run expects: the ranked job at index `picked`, or
    {"custom_role": role} when picked is CUSTOM. Raises ValueError if invalid."""
    if picked == CUSTOM:
        problem = role_problem(role)
        if problem:
            raise ValueError(problem)
        return {"custom_role": role.strip()}
    if not isinstance(picked, int) or not 0 <= picked < len(ranked):
        raise ValueError("pick one of the ranked jobs")
    return ranked[picked]


def choice_label(ranked, picked):
    """Radio label for a ranked job index or CUSTOM (escaped: untrusted text)."""
    if picked == CUSTOM:
        return "A different role (type it below)"
    job = ranked[picked]
    return (f"{picked + 1}. {md(job.get('title'), 80)} · "
            f"{md(job.get('company') or 'unknown company', 60)} "
            f"({float(job.get('score') or 0):.1f})")


def relative(path):
    return os.path.relpath(path, ROOT) if path and path.startswith(ROOT) else path


def breakdown_rows(breakdown):
    """Agent 3's score breakdown as (part, points, detail) rows."""
    b = breakdown or {}
    years = (f"needs {b.get('required_years', 0)} years, resume shows "
             f"{b.get('candidate_years', 0)}")
    degree = (f"needs {b.get('required_degree') or 'no degree'}, resume shows "
              f"{b.get('candidate_degree') or 'none'}")
    return [
        ("Skill overlap", b.get("skill_points", 0), "up to 60"),
        ("Text similarity", b.get("text_points", 0),
         f"up to 40 ({b.get('text_method', '?')}, similarity {b.get('text_similarity', 0)})"),
        ("Experience gap", -abs(b.get("experience_penalty", 0) or 0), years),
        ("Degree gap", -abs(b.get("degree_penalty", 0) or 0), degree),
    ]


# ---------- the page ----------

def job_card(rank, job):
    """One ranked job: score, title, company, skills, key points, gaps,
    interview focus, score breakdown and (if safe) the link."""
    a = job.get("assessment") or {}
    with st.container(border=True):
        st.markdown(f"#### {rank}. {md(job.get('title'))}")
        where = " · ".join(md(x) for x in (job.get("company") or "unknown company",
                                           job.get("location") or "location not given"))
        st.markdown(f"**Score {float(job.get('score') or 0):.1f} / 100** · {where}")
        left, right = st.columns(2)
        with left:
            st.markdown("**Matched skills**")
            st.markdown(bullets(job.get("matched_skills")))
            st.markdown("**Key points (what the employer wants)**")
            st.markdown(bullets(a.get("key_points")))
        with right:
            st.markdown("**Missing skills**")
            st.markdown(bullets(job.get("missing_skills")))
            st.markdown("**Gaps**")
            st.markdown(bullets(a.get("gaps")))
        st.markdown("**Interview focus (what Agent 4 will ask about)**")
        st.markdown(bullets(a.get("interview_focus")))
        with st.expander("Score breakdown"):
            st.table([{"Part": p, "Points": round(float(v or 0), 1), "Detail": d}
                      for p, v, d in breakdown_rows(job.get("breakdown"))])
        url = safe_url(job.get("url"))
        if url:
            st.link_button("Open the posting", url)
        else:
            st.caption("No safe link for this posting.")


def job_cards(ranked):
    st.subheader(f"{len(ranked)} ranked jobs, best match first")
    for i, job in enumerate(ranked, 1):
        job_card(i, job)


def choice_form(state):
    """The human-in-the-loop step: pick a job or type a role, then confirm
    (writes the Agent 4 brief) or quit (no brief)."""
    ranked = state["ranked_jobs"]
    st.subheader("2. Choose the job to practise for")
    options = list(range(len(ranked))) + [CUSTOM]
    picked = st.radio("Job", options, format_func=lambda p: choice_label(ranked, p),
                      key="picked", label_visibility="collapsed")
    role = ""
    if picked == CUSTOM:
        role = st.text_input("Role (e.g. Data Engineer)", max_chars=MAX_ROLE_LENGTH, key="role")
        st.caption("A custom role has no real posting: Agent 3 scores it from the role "
                   "name only, so its key points and gaps are thin.")
    confirm_col, quit_col = st.columns(2)
    confirm = confirm_col.button("Confirm and prepare the interview", type="primary")
    quit_ = quit_col.button("Quit without choosing")

    if quit_:
        finish(state, None)
    elif confirm:
        try:
            choice = make_choice(ranked, picked, role)
        except ValueError as e:
            st.error(str(e))
            return
        finish(state, choice)


def finish(state, choice):
    """Runs the second half once and keeps only a short summary."""
    message = ("Scoring your role and writing the interview brief..." if
               isinstance(choice, dict) and "custom_role" in choice else
               "Writing the interview brief and the run trace..." if choice else
               "Saving the run trace...")
    try:
        with st.spinner(message):
            result = finish_run(state, choice)
    except Exception as e:
        clear_run_state(st.session_state)      # a half-finished run can't be reused
        st.session_state["error"] = (f"Could not finish the run, please start over. "
                                     f"({type(e).__name__}: {e})")
        st.rerun()
    selected = result["selected"]
    st.session_state["result"] = {
        "title": selected["title"] if selected else None,
        "company": selected.get("company") if selected else None,
        "score": selected.get("score") if selected else None,
        "brief": relative(result["handoff"]["brief"]) if result["handoff"] else None,
        "trace": relative(result["trace_path"]),
    }
    clear_run_state(st.session_state)          # drop the resume text and the upload
    st.rerun()


def result_view(summary):
    if summary["brief"]:
        st.success(f"Ready for the interview: {md(summary['title'])} · "
                   f"{md(summary['company'] or 'custom role')} "
                   f"(score {float(summary['score'] or 0):.1f}).")
        st.markdown(f"Interview brief for Agent 4: `{summary['brief']}`")
    else:
        st.info("Stopped without choosing a job: no interview brief was written.")
    st.markdown(f"Run trace: `{summary['trace']}`")
    if st.button("Start over"):
        for key in ("result", "error"):
            st.session_state.pop(key, None)
        st.rerun()

def sidebar():
    """The inputs. Returns (uploaded file, location, mode, max_steps, pressed)."""
    with st.sidebar:
        st.header("1. Your resume")
        upload = st.file_uploader("Resume (PDF or DOCX)", type=list(ALLOWED_TYPES),
                                  key=f"resume_{st.session_state.get('upload_key', 0)}")
        st.caption("Your resume is only kept in memory: the temporary copy is deleted "
                   "as soon as it has been read, and everything is cleared once you "
                   "confirm or quit.")
        location = st.text_input("Location (optional)", placeholder="e.g. Monroe, LA or Remote")
        mode = st.radio("Mode", list(MODES), format_func=MODES.get, index=0)
        max_steps = LIVE_DEFAULT_STEPS
        if mode == "live":
            st.warning("Live mode spends JSearch requests (free plan: about 200 a month). "
                       "Each search step uses at most one request.")
            max_steps = st.number_input("Search steps (JSearch requests at most)",
                                        min_value=1, max_value=MAX_STEPS,
                                        value=LIVE_DEFAULT_STEPS, step=1)
        pressed = st.button("Find jobs", type="primary", disabled=upload is None)
    return upload, location, mode, max_steps, pressed


def main():
    st.set_page_config(page_title="INTERVAI - choose a job", layout="wide")
    st.title("INTERVAI: choose a job to practise for")
    upload, location, mode, max_steps, pressed = sidebar()

    if pressed and upload is not None:
        # The only place Agents 1-3 run: never on a plain rerun
        find_jobs(upload, location, mode, max_steps)

    if "error" in st.session_state:
        st.error(st.session_state["error"])
    if "result" in st.session_state:
        result_view(st.session_state["result"])
        return
    state = st.session_state.get("run")
    if state is None:
        if "error" not in st.session_state:
            st.info("Upload a resume, choose a mode and click **Find jobs**.")
        return
    st.caption(f"Jobs from: {md(state['jobs_source'])}")
    for level, message in stage_notices(state, st.session_state.get("mode")):
        getattr(st, level)(message)
    choice_form(state)
    job_cards(state["ranked_jobs"])


def find_jobs(upload, location, mode, max_steps):
    """Runs Agents 1-3 with a spinner per stage. On success the state goes
    to st.session_state["run"]; on failure a message to ["error"]."""
    for key in ("run", "result", "error"):
        st.session_state.pop(key, None)
    current = {"stage": None}
    with st.status("Finding jobs...", expanded=True) as status:
        def on_stage(key):
            current["stage"] = key
            status.update(label=stage_message(key, mode))
            st.write(stage_message(key, mode))
        try:
            state = run_agents(upload.name, upload.getvalue(), location, mode, max_steps,
                               on_stage=on_stage)
        except Exception as e:
            status.update(label="Finding jobs failed", state="error", expanded=False)
            st.session_state["error"] = failure_message(current["stage"], e)
            return
        status.update(label=f"Found {len(state['ranked_jobs'])} jobs", state="complete",
                      expanded=False)
    st.session_state["run"] = state
    st.session_state["mode"] = mode


if __name__ == "__main__":
    main()
