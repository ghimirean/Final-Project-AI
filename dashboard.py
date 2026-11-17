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
from orchestrator import MAX_STEPS, OUTPUT_DIR, prepare_run  # noqa: E402

ALLOWED_TYPES = ("pdf", "docx")
MODES = {
    "demo": "Demo: replay the last real jobs (no API calls)",
    "offline": "Offline: 6 fake practice jobs (no API calls)",
    "live": "Live: search JSearch now (spends requests)",
}
LIVE_DEFAULT_STEPS = 1


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


def run_agents(filename, data, location, mode, max_steps=LIVE_DEFAULT_STEPS,
               prepare=prepare_run):
    """Writes the uploaded resume to a temp file, runs Agents 1-3 on it and
    deletes the file again, even if an agent fails. Returns the run state."""
    options = mode_options(mode, max_steps)
    ext = file_extension(filename)
    fd, path = tempfile.mkstemp(suffix=ext, prefix="intervai_resume_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return prepare(path, (location or "").strip(), out_dir=output_dir(), **options)
    finally:
        os.remove(path)


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

def sidebar():
    """The inputs. Returns (uploaded file, location, mode, max_steps, pressed)."""
    with st.sidebar:
        st.header("1. Your resume")
        upload = st.file_uploader("Resume (PDF or DOCX)", type=list(ALLOWED_TYPES))
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
        state = run_agents(upload.name, upload.getvalue(), location, mode, max_steps)
        st.session_state["run"] = state
        st.session_state["mode"] = mode

    state = st.session_state.get("run")
    if state is None:
        st.info("Upload a resume, choose a mode and click **Find jobs**.")
        return
    st.caption(f"Jobs from: {md(state['jobs_source'])}")
    job_cards(state["ranked_jobs"])


if __name__ == "__main__":
    main()
