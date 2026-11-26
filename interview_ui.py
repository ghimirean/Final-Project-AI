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
from interview_agent import MAX_ANSWER_SECONDS, OUTPUT_DIR, build_interview  # noqa: E402
from interview_questions import BriefError  # noqa: E402
from interview_recorder import AnswerRecorder, new_recordings_dir  # noqa: E402

BRIEF_NAME = "interview_brief.json"


# ---------- non-UI logic (plain functions, tested without Streamlit) ----------

def output_dir():
    """Where the brief is: output/ (like the rest of INTERVAI), or the folder
    in INTERVAI_OUTPUT_DIR (the tests use a temp folder)."""
    return os.environ.get("INTERVAI_OUTPUT_DIR") or OUTPUT_DIR


def brief_path():
    return os.path.join(output_dir(), BRIEF_NAME)


def recordings_dir():
    """This interview's folder: recordings/<date-time>/ like Agent 4's, under the
    folder in INTERVAI_RECORDINGS_DIR if set (the tests use a temp folder)."""
    base = os.environ.get("INTERVAI_RECORDINGS_DIR")
    return new_recordings_dir(base) if base else new_recordings_dir()


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


# One question goes through three phases: "ready" -> (Start answering)
# "recording" -> (Stop) "stopped" -> (Next question / Finish) the next one.
READY, RECORDING, STOPPED = "ready", "recording", "stopped"


def button_states(phase):
    """Which buttons can be clicked in a phase: {"start", "stop", "next"} -> enabled."""
    return {"start": phase == READY, "stop": phase == RECORDING, "next": phase == STOPPED}


def new_session(interview, folder=None):
    """The st.session_state values for a freshly started interview."""
    return {"interview": interview, "started": True, "index": 0, "phase": READY,
            "recorder": None, "answers": [], "finished": False,
            "recordings_dir": folder or recordings_dir()}


def device_problems(errors):
    """Agent 4's error strings ("speech: ...", "vision: ...") in words for the user."""
    names = {"speech": "Microphone", "vision": "Webcam"}
    out = []
    for error in errors:
        kind, _, text = str(error).partition(": ")
        out.append(f"{names.get(kind, kind)}: {text}" if text else str(error))
    return out


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
        st.session_state.update(new_session(interview))
        st.rerun()


def question_page():
    """One question at a time: Start answering -> Stop -> Next question / Finish."""
    s = st.session_state
    questions = s["interview"]["questions"]
    index, phase = s["index"], s["phase"]
    question = questions[index]
    last = index == len(questions) - 1

    st.caption(f"Question {index + 1} of {len(questions)}")
    st.subheader(md(question["question"], limit=1000))   # Agent 4's template + topics from the posting

    recorder = s.get("recorder")
    if phase == RECORDING:
        if recorder is not None and recorder.recording:
            st.warning(f"Recording (up to {MAX_ANSWER_SECONDS} s). Look at the camera and answer out "
                       "loud, then click **Stop**.")
        else:
            st.warning("The recording has ended (time limit reached or no device could record). "
                       "Click **Stop** to continue.")
    elif phase == STOPPED:
        got = s["answers"][index]
        problems = device_problems(got["errors"])
        if got["audio"] or got["video"]:
            st.success("Answer recorded.")
        else:
            st.error("Nothing could be recorded for this question.")
        for problem in problems:
            st.caption(problem)

    enabled = button_states(phase)
    start_col, stop_col, next_col = st.columns(3)
    if start_col.button("Start answering", type="primary", disabled=not enabled["start"]):
        recorder = AnswerRecorder(question["id"], s["recordings_dir"])
        recorder.start()
        s["recorder"], s["phase"] = recorder, RECORDING
        st.rerun()
    if stop_col.button("Stop", disabled=not enabled["stop"]):
        got = s["recorder"].stop()
        s["answers"].append({"id": question["id"], **got})
        s["recorder"], s["phase"] = None, STOPPED
        st.rerun()
    if next_col.button("Finish" if last else "Next question", disabled=not enabled["next"]):
        if last:
            s["finished"] = True
        else:
            s["index"], s["phase"] = index + 1, READY
        st.rerun()


def finished_page():
    answers = st.session_state["answers"]
    recorded = sum(1 for a in answers if a["audio"] or a["video"])
    st.success(f"Interview finished: {recorded} of {len(answers)} answers recorded.")
    st.caption("The answers are saved on this computer. Analysing them comes in a later step.")


def main():
    st.set_page_config(page_title="INTERVAI - mock interview", layout="centered")
    st.title("INTERVAI: mock interview")
    if st.session_state.get("finished"):
        finished_page()
        return
    if st.session_state.get("started"):
        question_page()
        return
    try:
        interview = load_interview()
    except BriefError as e:
        no_brief(e)
        return
    start_page(interview)


if __name__ == "__main__":
    main()
