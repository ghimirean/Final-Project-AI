"""INTERVAI mock interview page: Agent 4's front end, in the browser.

    streamlit run interview_ui.py

Reads output/interview_brief.json (written when a job is chosen in the
dashboard or the orchestrator) and builds the questions with Agent 4's own
functions (agents/interview_agent.py, interview_questions.py). This page is
only a front end: no question, recording or analysis logic is duplicated.

Streamlit reruns this whole script on every click, so everything about the
interview lives in st.session_state.
"""
import os
import re
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "agents"))
from interview_agent import OUTPUT_DIR, build_interview  # noqa: E402
from interview_questions import BriefError  # noqa: E402

BRIEF_NAME = "interview_brief.json"


# ---------- non-UI logic (plain functions, tested without Streamlit) ----------

def output_dir():
    """Where the brief is: output/ (like the rest of INTERVAI), or the folder
    in INTERVAI_OUTPUT_DIR (the tests use a temp folder)."""
    return os.environ.get("INTERVAI_OUTPUT_DIR") or OUTPUT_DIR


def brief_path():
    return os.path.join(output_dir(), BRIEF_NAME)


def load_interview(path=None):
    """Agent 4's questions for the current brief: {"brief", "job", "questions"}.
    Raises BriefError (missing, stale or wrong version) with a message for the user."""
    return build_interview(path or brief_path())


# The job title and company come from a job posting on the internet, so they
# are untrusted: no HTML is ever rendered (never unsafe_allow_html), and
# markdown characters are escaped (same rule as dashboard.py).
MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:=&\"'^])")


def md(text, limit=200):
    """Untrusted text -> safe for st.markdown: one line, at most `limit`
    characters, every markdown character escaped."""
    text = " ".join(str(text if text is not None else "").split())
    if len(text) > limit:
        text = text[:limit - 1].rstrip() + "…"
    return MARKDOWN_SPECIAL.sub(r"\\\1", text)


def job_line(job):
    """'Junior Java Developer at Maplewood Software' (escaped)."""
    title = md(job.get("title") or "this role")
    return f"{title} at {md(job['company'])}" if job.get("company") else title


# ---------- the page ----------

def no_brief(error):
    st.error("There is no interview to start yet.\n\n"
             f"{error}\n\n"
             "Choose a job first: in the dashboard (`streamlit run dashboard.py`) click "
             "**Confirm and prepare the interview**, or run the orchestrator and pick a job. "
             "Then reload this page.")


def start_page(interview):
    job, questions = interview["job"], interview["questions"]
    st.subheader(f"Mock interview: {job_line(job)}")
    st.markdown(f"**{len(questions)} questions.** For each one you will start and stop the "
                "recording yourself, and answer out loud while looking at the camera.")
    st.info("Your answers (microphone and webcam) are recorded and analysed on this computer "
            "only. Nothing is uploaded.")
    if st.button("Start the interview", type="primary"):
        st.session_state["interview"] = interview
        st.session_state["started"] = True
        st.rerun()


def main():
    st.set_page_config(page_title="INTERVAI - mock interview", layout="centered")
    st.title("INTERVAI: mock interview")
    if st.session_state.get("started"):
        st.info("The interview has started.")
        return
    try:
        interview = load_interview()
    except BriefError as e:
        no_brief(e)
        return
    start_page(interview)


if __name__ == "__main__":
    main()
