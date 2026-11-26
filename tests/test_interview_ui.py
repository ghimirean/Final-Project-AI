"""Tests for the mock interview page (interview_ui.py), with Streamlit's AppTest.

Everything is FAKE: the microphone and webcam (interview_speech.record_audio,
interview_vision.record_video) and the analysers (Whisper, MediaPipe) are
replaced, so nothing is recorded or analysed for real. AppTest runs the page
in this process, so the fakes apply to it. The brief, results and recordings
live in temp folders (INTERVAI_OUTPUT_DIR, INTERVAI_RECORDINGS_DIR): the real
output/ and recordings/ are never touched. All data is fake.
"""
import json
import os
import time

import pytest
from streamlit.testing.v1 import AppTest

import interview_speech
import interview_vision
from interview_speech import SpeechError
from interview_vision import VisionError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "interview_ui.py")

BRIEF = {"brief_version": 1, "created": "2026-11-25T10:00:00", "jobs_source": "offline",
         "job": {"id": "job-1", "title": "Data Analyst", "company": "Acme Analytics", "gaps": ["Linux"],
                 "interview_focus": ["Linux (asked for, not on the resume)"]},
         "candidate": {"skills": ["Python"], "projects": [{"name": "Sales Dashboard", "role": "Lead"}]}}
TRANSCRIPT = "I used <b>Python</b> and [a link](javascript:alert(1)) at work"
SPEECH = {"duration_seconds": 4.0, "speaking_seconds": 4.0, "word_count": 10, "words_per_minute": 150.0,
          "filler_count": 1, "filler_rate_per_100_words": 10.0, "fillers": {"um": 1}, "long_pause_count": 0,
          "long_pauses": [], "seconds_before_first_word": 0.5, "transcript": TRANSCRIPT}
VISION = {"duration_seconds": 2.0, "frames_analyzed": 10, "face_detected_percent": 100.0,
          "eye_contact_percent": 70.0, "longest_look_away_seconds": 1.0, "pose_detected_percent": 0.0,
          "posture_upright_percent": None, "average_shoulder_tilt_degrees": None, "smile_percent": 5.0}


def wait_for(stop, limit=3.0):
    end = time.time() + limit
    while not stop() and time.time() < end:
        time.sleep(0.01)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Temp folders + fake devices and analysers. Returns the folders and a call log."""
    out, rec = tmp_path / "output", tmp_path / "recordings"
    out.mkdir()
    monkeypatch.setenv("INTERVAI_OUTPUT_DIR", str(out))
    monkeypatch.setenv("INTERVAI_RECORDINGS_DIR", str(rec))
    calls = {"analyze_audio": 0, "analyze_video": 0}

    def record_audio(path, max_seconds=0, stop=None):
        wait_for(stop)
        open(path, "wb").write(b"fake audio")

    def record_video(path, stop=None, max_seconds=0, camera=None):
        wait_for(stop)
        open(path, "wb").write(b"fake video")

    def analyze_audio(path):
        calls["analyze_audio"] += 1
        return dict(SPEECH)

    def analyze_video(path):
        calls["analyze_video"] += 1
        return dict(VISION)
    monkeypatch.setattr(interview_speech, "record_audio", record_audio)
    monkeypatch.setattr(interview_vision, "record_video", record_video)
    monkeypatch.setattr(interview_speech, "analyze_audio", analyze_audio)
    monkeypatch.setattr(interview_vision, "analyze_video", analyze_video)
    return {"out": out, "rec": rec, "calls": calls}


def write_brief(env, **changes):
    (env["out"] / "interview_brief.json").write_text(json.dumps(dict(BRIEF, **changes)), encoding="utf-8")


def new_app():
    return AppTest.from_file(APP, default_timeout=60).run()


def button(at, label):
    return next(b for b in at.button if b.label == label)


def start(at, keep=False):
    if keep:
        at.checkbox(key="keep_choice").check().run()
    button(at, "Start the interview").click().run()
    return at


def answer_all(at):
    total = len(at.session_state["interview"]["questions"])
    for i in range(total):
        button(at, "Start answering").click().run()
        button(at, "Stop").click().run()
        button(at, "Finish" if i == total - 1 else "Next question").click().run()
    return at


def results_file(env):
    with open(env["out"] / "interview_results.json", encoding="utf-8") as f:
        return json.load(f)


def metrics(at):
    return {m.label: m.value for m in at.metric[:6]}


# ---------- the start page ----------

def test_no_brief_says_to_choose_a_job_first(env):
    at = new_app()
    assert not at.exception
    assert "Choose a job first" in at.error[0].value
    assert not at.button                                     # nothing to start


def test_a_wrong_version_brief_is_refused(env):
    write_brief(env, brief_version=99)
    at = new_app()
    assert "version 99" in at.error[0].value and "Choose a job first" in at.error[0].value
    assert not at.button


def test_start_page_shows_the_job_and_the_question_count(env):
    write_brief(env)
    at = new_app()
    assert "Data Analyst at Acme Analytics" in at.subheader[0].value
    assert "questions" in at.markdown[0].value
    assert at.checkbox(key="keep_choice").value is False     # recordings are deleted by default
    assert any("this computer only" in i.value for i in at.info)


def test_starting_removes_old_results(env):
    write_brief(env)
    (env["out"] / "interview_results.json").write_text('{"old": true}', encoding="utf-8")
    start(new_app())
    assert not (env["out"] / "interview_results.json").exists()


# ---------- one question at a time ----------

def test_buttons_follow_the_phases(env):
    write_brief(env)
    at = start(new_app())
    state = lambda: {b.label: b.disabled for b in at.button}
    assert at.caption[0].value.startswith("Question 1 of ")
    assert state() == {"Start answering": False, "Stop": True, "Next question": True}
    button(at, "Start answering").click().run()
    assert state() == {"Start answering": True, "Stop": False, "Next question": True}
    button(at, "Stop").click().run()
    assert state() == {"Start answering": True, "Stop": True, "Next question": False}
    assert any(s.value == "Answer recorded." for s in at.success)


def test_all_questions_answered_gives_a_scorecard(env):
    write_brief(env)
    at = answer_all(start(new_app()))
    assert not at.exception
    total = len(at.session_state["interview"]["questions"])
    assert any(s.value == f"Analysed {total} of {total} answers." for s in at.success)
    assert at.status[0].label == "Analysis finished"
    assert [m.value for m in at.markdown if m.value.startswith("Analysing answer")] == \
        [f"Analysing answer {n} of {total}..." for n in range(1, total + 1)]
    assert metrics(at)["Words per minute"] == "150" and metrics(at)["Eye contact"] == "70%"
    assert [e.label for e in at.expander] == [f"Question {n}" for n in range(1, total + 1)]
    assert results_file(env)["overall"]["questions_answered"] == total
    assert any("not a trained classifier" in c.value for c in at.caption)


def test_missing_values_show_as_n_a(env):
    write_brief(env)
    at = answer_all(start(new_app()))
    assert metrics(at)["Upright posture"] == "n/a"             # posture None: the body was never visible


def test_a_failing_webcam_on_one_question(env, monkeypatch):
    real = interview_vision.record_video

    def camera_fails_on_q2(path, stop=None, max_seconds=0, camera=None):
        if path.endswith("q2.avi"):
            raise VisionError("Could not open the webcam.")
        return real(path, stop=stop, max_seconds=max_seconds)
    monkeypatch.setattr(interview_vision, "record_video", camera_fails_on_q2)
    write_brief(env)
    at = start(new_app())
    button(at, "Start answering").click().run()
    button(at, "Stop").click().run()
    button(at, "Next question").click().run()
    button(at, "Start answering").click().run()
    button(at, "Stop").click().run()
    assert any(c.value == "Video (webcam): Could not open the webcam." for c in at.caption)
    assert any(s.value == "Answer recorded." for s in at.success)      # the microphone still worked
    answer_rest(at)
    assert any(c.value == "Question 2: Video (webcam): Could not open the webcam." for c in at.caption)
    q2 = results_file(env)["questions"][1]
    assert q2["answered"] and q2["speech"] and q2["vision"] is None


def answer_rest(at):
    """From a stopped question to the end."""
    while True:
        labels = {b.label for b in at.button}
        if "Finish" in labels:
            button(at, "Finish").click().run()
            return
        button(at, "Next question").click().run()
        button(at, "Start answering").click().run()
        button(at, "Stop").click().run()


def test_nothing_analysed_shows_no_scorecard(env, monkeypatch):
    def no_mic(path, max_seconds=0, stop=None):
        raise SpeechError("no microphone")

    def no_cam(path, stop=None, max_seconds=0, camera=None):
        raise VisionError("no webcam")
    monkeypatch.setattr(interview_speech, "record_audio", no_mic)
    monkeypatch.setattr(interview_vision, "record_video", no_cam)
    write_brief(env)
    at = answer_all(start(new_app()))
    assert any("None of your answers could be analysed" in e.value for e in at.error)
    assert not at.metric and not at.expander


# ---------- recordings: deleted by default, kept on request ----------

def test_recordings_are_deleted_after_analysis_by_default(env):
    write_brief(env)
    at = answer_all(start(new_app()))
    assert not env["rec"].exists() or os.listdir(env["rec"]) == []
    assert all(q["recordings"] == {"audio": None, "video": None} for q in results_file(env)["questions"])
    assert any(c.value == "Recordings deleted." for c in at.caption)


def test_keep_my_recordings_keeps_them(env):
    write_brief(env)
    at = answer_all(start(new_app(), keep=True))
    folders = os.listdir(env["rec"])
    assert len(folders) == 1
    files = sorted(os.listdir(env["rec"] / folders[0]))
    total = len(at.session_state["interview"]["questions"])
    assert files == sorted(f"q{n}.{ext}" for n in range(1, total + 1) for ext in ("wav", "avi"))
    assert results_file(env)["questions"][0]["recordings"]["audio"].endswith("q1.wav")
    assert any(c.value.startswith("Recordings kept in ") for c in at.caption)


def test_no_repeated_analysis_on_a_rerun(env):
    write_brief(env)
    at = answer_all(start(new_app()))
    before = dict(env["calls"])
    at.run()
    at.run()
    assert env["calls"] == before
    assert before["analyze_audio"] == len(at.session_state["interview"]["questions"])


# ---------- untrusted text ----------

def test_html_and_javascript_links_are_plain_text(env):
    write_brief(env, job=dict(BRIEF["job"], title="<b>Data</b> [Analyst](javascript:alert(1))"))
    at = new_app()
    assert "\\<b\\>Data\\</b\\>" in at.subheader[0].value              # escaped, not rendered
    assert "\\[Analyst\\]\\(javascript\\:alert\\(1\\)\\)" in at.subheader[0].value
    at = answer_all(start(at))
    texts = [t.value for t in at.text]
    assert TRANSCRIPT in texts                                         # st.text: never HTML or markdown
    assert all(TRANSCRIPT not in m.value for m in at.markdown)


# ---------- interview again ----------

def test_interview_again_goes_back_to_the_start_page(env):
    write_brief(env)
    at = answer_all(start(new_app()))
    button(at, "Interview again").click().run()
    assert [b.label for b in at.button] == ["Start the interview"]
    assert "interview" not in at.session_state and "results" not in at.session_state
    assert (env["out"] / "interview_results.json").exists()           # the last results stay in the file
