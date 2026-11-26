"""Tests for Agent 4, step 4b: the whole interview (run_interview in agents/interview_agent.py).
Offline: the microphone, webcam, Whisper and MediaPipe are fakes; the brief and the audio are
synthetic. Output goes to a temp folder. All data is fake."""
import json
import os
import threading
import time

import numpy as np
import pytest

import interview_agent
import interview_speech
import interview_vision
from interview_agent import InterviewError, run_interview
from interview_questions import BriefError
from interview_speech import SpeechError
from interview_vision import VisionError


def make_brief(path, created="2026-11-18T10:00:00"):
    brief = {"brief_version": 1, "created": created, "jobs_source": "offline",
             "job": {"id": "job-1", "title": "Junior Java Developer", "company": "Maplewood Software",
                     "gaps": ["Linux"], "interview_focus": ["Linux (asked for, not on the resume)",
                                                            "Java (ask for an example; resume: skills: Java)"]},
             "candidate": {"projects": [{"name": "Car Rental System", "role": "Lead Developer"}]}}
    path.write_text(json.dumps(brief), encoding="utf-8")
    return str(path)


@pytest.fixture
def brief(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    return make_brief(out / "interview_brief.json")


def tone(seconds):
    t = np.arange(int(seconds * interview_speech.SAMPLE_RATE)) / interview_speech.SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


WORDS = [{"word": w, "start": i * 0.5, "end": i * 0.5 + 0.4}
         for i, w in enumerate("I built a tool, um, with Java and Git for two years".split())]
VISION = {"duration_seconds": 3.0, "frames_analyzed": 15, "face_detected_percent": 100.0, "eye_contact_percent": 80.0,
          "longest_look_away_seconds": 0.6, "pose_detected_percent": 100.0, "posture_upright_percent": 90.0,
          "average_shoulder_tilt_degrees": 3.0, "smile_percent": 10.0}


@pytest.fixture(autouse=True)
def quick_device_check(monkeypatch):
    monkeypatch.setattr(interview_agent, "DEVICE_START_GRACE", 0.02)


@pytest.fixture
def devices(monkeypatch):
    """Fake microphone, webcam, Whisper and MediaPipe. Returns a log of what was called."""
    log = {"audio": [], "video": []}

    def wait_for(stop, limit=3.0):
        end = time.time() + limit
        while not stop() and time.time() < end:
            time.sleep(0.01)

    def record_audio(path, max_seconds=0, stop=None):
        log["audio"].append((os.path.basename(path), max_seconds))
        wait_for(stop)
        interview_speech.write_wav(path, tone(1.0))
        return 1.0

    def record_video(path, stop=None, max_seconds=0, camera=None):
        log["video"].append((os.path.basename(path), max_seconds))
        wait_for(stop)
        open(path, "wb").write(b"fake video")
        return 1.0
    monkeypatch.setattr(interview_speech, "record_audio", record_audio)
    monkeypatch.setattr(interview_vision, "record_video", record_video)
    monkeypatch.setattr(interview_speech, "transcribe", lambda samples: WORDS)
    monkeypatch.setattr(interview_vision, "analyze_video", lambda path: dict(VISION))
    return log


def enter_log():
    prompts = []
    return prompts, (lambda text="": prompts.append(text))


# ---------- live interview ----------

def test_live_interview_end_to_end(brief, tmp_path, devices):
    prompts, press_enter = enter_log()
    said = []
    results = run_interview(brief, recordings_dir=str(tmp_path / "rec"), wait_for_enter=press_enter, say=said.append,
                            keep_recordings=True)
    n = results["overall"]["questions_total"]
    assert 4 <= n <= 5 and results["overall"]["questions_answered"] == n and results["mode"] == "live"
    assert len(prompts) == 2 * n                                     # one Enter to start, one to stop, per question
    first = results["questions"][0]
    assert first["kind"] == "intro" and first["answered"] and first["errors"] == []
    assert first["speech"]["word_count"] == len(WORDS) and first["speech"]["fillers"] == {"um": 1}
    assert first["vision"]["eye_contact_percent"] == 80.0
    assert first["answer_seconds"] == pytest.approx(max(first["speech"]["duration_seconds"], 3.0))
    assert os.path.isfile(tmp_path / "rec" / "q1.wav") and os.path.isfile(tmp_path / "rec" / "q1.avi")
    assert first["recordings"]["audio"].endswith("q1.wav")
    assert results["overall"]["speech"]["total_words"] == len(WORDS) * n
    assert results["overall"]["vision"]["eye_contact_percent"] == 80.0
    assert any("Question 1 of" in line for line in said)
    saved = json.load(open(os.path.join(os.path.dirname(brief), "interview_results.json"), encoding="utf-8"))
    assert saved == json.loads(json.dumps(results))                   # the file is exactly what is returned


def test_results_contain_no_name_or_contact_details_from_the_brief(brief, tmp_path, devices):
    results = run_interview(brief, recordings_dir=str(tmp_path / "rec"), wait_for_enter=enter_log()[1], say=lambda *a: None)
    assert set(results) == {"results_version", "created", "mode", "brief", "job", "questions", "overall", "settings"}
    assert set(results["job"]) == {"id", "title", "company"}


def test_microphone_and_webcam_record_at_the_same_time(brief, tmp_path, monkeypatch, devices):
    barrier = threading.Barrier(2, timeout=3)                        # passes only if both are running together
    real_audio, real_video = interview_speech.record_audio, interview_vision.record_video

    def audio(path, max_seconds=0, stop=None):
        barrier.wait()
        return real_audio(path, max_seconds, stop)

    def video(path, stop=None, max_seconds=0, camera=None):
        barrier.wait()
        return real_video(path, stop, max_seconds)
    monkeypatch.setattr(interview_speech, "record_audio", audio)
    monkeypatch.setattr(interview_vision, "record_video", video)
    results = run_interview(brief, recordings_dir=str(tmp_path / "rec"), wait_for_enter=enter_log()[1], say=lambda *a: None)
    assert results["overall"]["questions_answered"] == results["overall"]["questions_total"]
    barrier.reset()


def test_the_recorders_get_the_time_limit_and_stop_on_enter(brief, tmp_path, devices):
    run_interview(brief, recordings_dir=str(tmp_path / "rec"), wait_for_enter=enter_log()[1], say=lambda *a: None,
                  max_answer_seconds=45)
    assert devices["audio"][0] == ("q1.wav", 45) and devices["video"][0] == ("q1.avi", 45)


def test_webcam_failure_keeps_the_speech_results(brief, tmp_path, monkeypatch, devices):
    def no_camera(path, stop=None, max_seconds=0, camera=None):
        raise VisionError("Could not open the webcam.")
    monkeypatch.setattr(interview_vision, "record_video", no_camera)
    results = run_interview(brief, recordings_dir=str(tmp_path / "rec"), wait_for_enter=enter_log()[1], say=lambda *a: None)
    q = results["questions"][0]
    assert q["answered"] and q["speech"] and q["vision"] is None
    assert q["errors"] == ["vision: Could not open the webcam."]
    assert results["overall"]["vision"] is None and results["overall"]["speech"]["total_words"] > 0


def test_both_devices_failing_does_not_ask_for_enter_and_still_writes_results(brief, tmp_path, monkeypatch, devices):
    def mic_error(path, max_seconds=0, stop=None):
        raise SpeechError("no microphone")

    def cam_error(path, stop=None, max_seconds=0, camera=None):
        raise VisionError("no webcam")
    monkeypatch.setattr(interview_speech, "record_audio", mic_error)
    monkeypatch.setattr(interview_vision, "record_video", cam_error)
    prompts, press_enter = enter_log()
    said = []
    results = run_interview(brief, recordings_dir=str(tmp_path / "rec"), wait_for_enter=press_enter, say=said.append)
    n = results["overall"]["questions_total"]
    assert len(prompts) == n                                           # only the "start" Enter, never the "stop" one
    assert results["overall"]["questions_answered"] == 0 and results["overall"]["speech"] is None
    assert results["questions"][0]["errors"] == ["speech: no microphone", "vision: no webcam"]
    assert any("Nothing could be recorded" in line for line in said)
    assert os.path.isfile(os.path.join(os.path.dirname(brief), "interview_results.json"))


def test_an_unexpected_device_crash_is_recorded_not_raised(brief, tmp_path, monkeypatch, devices):
    def crash(path, stop=None, max_seconds=0, camera=None):
        raise RuntimeError("driver exploded")
    monkeypatch.setattr(interview_vision, "record_video", crash)
    results = run_interview(brief, recordings_dir=str(tmp_path / "rec"), wait_for_enter=enter_log()[1], say=lambda *a: None)
    assert "vision: unexpected recording error: driver exploded" in results["questions"][0]["errors"]


def test_whisper_failure_is_recorded_and_vision_is_kept(brief, tmp_path, monkeypatch, devices):
    def broken(path):
        raise SpeechError("Could not load the Whisper model")
    monkeypatch.setattr(interview_speech, "analyze_audio", broken)
    results = run_interview(brief, recordings_dir=str(tmp_path / "rec"), wait_for_enter=enter_log()[1], say=lambda *a: None)
    q = results["questions"][0]
    assert q["speech"] is None and q["vision"] and q["answered"]
    assert q["errors"] == ["speech: Could not load the Whisper model"]


# ---------- pre-recorded files ----------

def test_files_mode_uses_the_given_files_and_never_records(brief, tmp_path, monkeypatch, devices):
    audio = interview_speech.write_wav(str(tmp_path / "a1.wav"), tone(2.0))
    video = tmp_path / "v1.avi"
    video.write_bytes(b"x")
    prompts, press_enter = enter_log()
    results = run_interview(brief, audio_files=[audio], video_files=[str(video)], wait_for_enter=press_enter, say=lambda *a: None)
    assert results["mode"] == "files" and prompts == [] and devices == {"audio": [], "video": []}
    assert results["overall"]["questions_answered"] == 1 and results["overall"]["questions_total"] >= 4
    first, second = results["questions"][0], results["questions"][1]
    assert first["answered"] and first["speech"]["duration_seconds"] == 2.0 and first["vision"]
    assert second["answered"] is False and second["speech"] is None and second["errors"] == []
    assert os.path.isfile(audio)                                      # the given files are never deleted or moved


def test_files_mode_audio_only_and_video_only(brief, tmp_path, devices):
    audio = interview_speech.write_wav(str(tmp_path / "a.wav"), tone(1.0))
    only_audio = run_interview(brief, audio_files=[audio], say=lambda *a: None)
    assert only_audio["questions"][0]["vision"] is None and only_audio["overall"]["vision"] is None
    video = tmp_path / "v.avi"
    video.write_bytes(b"x")
    only_video = run_interview(brief, video_files=[str(video)], say=lambda *a: None)
    assert only_video["questions"][0]["speech"] is None and only_video["overall"]["speech"] is None


def test_none_skips_a_question_in_the_file_lists(brief, tmp_path, devices):
    audio = interview_speech.write_wav(str(tmp_path / "a.wav"), tone(1.0))
    results = run_interview(brief, audio_files=[None, audio], say=lambda *a: None)
    assert [q["answered"] for q in results["questions"][:3]] == [False, True, False]


def test_a_missing_answer_file_is_an_error_on_that_question(brief, tmp_path, devices):
    results = run_interview(brief, audio_files=[str(tmp_path / "gone.wav")], say=lambda *a: None)
    assert results["questions"][0]["answered"] is False
    assert "speech: Audio file not found" in results["questions"][0]["errors"][0]


def test_more_files_than_questions(brief, tmp_path, devices):
    with pytest.raises(InterviewError, match="answer files"):
        run_interview(brief, audio_files=["a.wav"] * 9)


# ---------- the brief ----------

def test_no_brief_means_no_interview_and_no_old_results(tmp_path, devices):
    out = tmp_path / "output"
    out.mkdir()
    (out / "interview_results.json").write_text("{}", encoding="utf-8")
    with pytest.raises(BriefError):
        run_interview(str(out / "interview_brief.json"))
    assert not (out / "interview_results.json").exists()            # no old results left to be mistaken for new ones


def test_a_stale_brief_is_refused(tmp_path, devices):
    out = tmp_path / "output"
    out.mkdir()
    (out / "run_trace.json").write_text(json.dumps({"started": "2026-11-18T12:00:00"}), encoding="utf-8")
    with pytest.raises(BriefError, match="earlier run"):
        run_interview(make_brief(out / "interview_brief.json"))


def test_old_results_are_replaced(brief, tmp_path, devices):
    old = os.path.join(os.path.dirname(brief), "interview_results.json")
    open(old, "w").write('{"old": true}')
    run_interview(brief, recordings_dir=str(tmp_path / "rec"), wait_for_enter=enter_log()[1], say=lambda *a: None)
    assert "old" not in json.load(open(old))


# ---------- command line ----------

def test_command_line_questions_only(brief, capsys):
    assert interview_agent.main(["--brief", brief, "--questions-only"]) == 0
    assert "Q1 [intro]" in capsys.readouterr().out


def test_command_line_with_files_prints_a_summary(brief, tmp_path, devices, capsys):
    audio = interview_speech.write_wav(str(tmp_path / "a.wav"), tone(2.0))
    assert interview_agent.main(["--brief", brief, "--audio", audio]) == 0
    out = capsys.readouterr().out
    assert "Answered 1 of" in out and "Speech:" in out and "filler words" in out
    assert os.path.isfile(os.path.join(os.path.dirname(brief), "interview_results.json"))


def test_command_line_errors_exit_with_1(tmp_path, brief, capsys):
    assert interview_agent.main(["--brief", str(tmp_path / "missing.json")]) == 1
    assert "Agent 4 cannot start" in capsys.readouterr().err
    assert interview_agent.main(["--brief", brief, "--audio"] + ["a.wav"] * 9) == 1
    assert "answer files" in capsys.readouterr().err


def test_command_line_exits_1_when_no_answer_could_be_analysed(brief, tmp_path, monkeypatch, capsys):
    audio = interview_speech.write_wav(str(tmp_path / "a.wav"), tone(1.0))
    assert interview_agent.main(["--brief", brief, "--audio", audio]) == 1          # conftest: Whisper is unavailable
    out = capsys.readouterr().out
    assert "Answered 0 of" in out and "Question 1: speech:" in out


# ---------- recordings: deleted after analysis unless kept ----------

def results_file(brief):
    with open(os.path.join(os.path.dirname(brief), "interview_results.json"), encoding="utf-8") as f:
        return json.load(f)


def test_live_recordings_are_deleted_after_analysis_by_default(brief, tmp_path, devices):
    rec = tmp_path / "rec"
    said = []
    results = run_interview(brief, recordings_dir=str(rec), wait_for_enter=enter_log()[1], say=said.append)
    n = results["overall"]["questions_total"]
    assert results["overall"]["questions_answered"] == n          # analysed first ...
    assert results["questions"][0]["speech"]["word_count"] == len(WORDS)
    assert not rec.exists()                                       # ... then files and the empty folder are gone
    assert said[-1] == "Recordings deleted."


def test_results_paths_are_null_after_deletion(brief, tmp_path, devices):
    results = run_interview(brief, recordings_dir=str(tmp_path / "rec"), wait_for_enter=enter_log()[1],
                            say=lambda *a: None)
    for saved in (results, results_file(brief)):                   # the returned dict and the file agree
        assert all(q["recordings"] == {"audio": None, "video": None} for q in saved["questions"])


def test_keep_recordings_keeps_files_and_paths(brief, tmp_path, devices):
    rec = tmp_path / "rec"
    said = []
    results = run_interview(brief, recordings_dir=str(rec), wait_for_enter=enter_log()[1], say=said.append,
                            keep_recordings=True)
    n = results["overall"]["questions_total"]
    assert sorted(os.listdir(rec)) == sorted([f"q{i}.{e}" for i in range(1, n + 1) for e in ("wav", "avi")])
    first = results_file(brief)["questions"][0]["recordings"]
    assert first["audio"].endswith("q1.wav") and first["video"].endswith("q1.avi")
    assert said[-1].startswith("Recordings kept in ") and said[-1].endswith("rec")


def test_pre_recorded_files_are_never_deleted(brief, tmp_path, devices):
    audio = interview_speech.write_wav(str(tmp_path / "a1.wav"), tone(2.0))
    video = tmp_path / "v1.avi"
    video.write_bytes(b"x")
    said = []
    results = run_interview(brief, audio_files=[audio], video_files=[str(video)], say=said.append)
    assert results["mode"] == "files"
    assert os.path.isfile(audio) and video.read_bytes() == b"x"
    assert results["questions"][0]["recordings"]["audio"] is not None   # their paths stay in the results
    assert not any("Recordings" in line for line in said)


def test_a_failed_run_does_not_delete_files_it_did_not_create(brief, tmp_path, monkeypatch, devices):
    rec = tmp_path / "rec"
    rec.mkdir()
    (rec / "notes.txt").write_text("mine", encoding="utf-8")       # someone else's file in the folder
    (rec / "q1.wav").write_bytes(b"old")                          # left from before; this run fails to record q1

    def mic_error(path, max_seconds=0, stop=None):
        raise SpeechError("no microphone")
    monkeypatch.setattr(interview_speech, "record_audio", mic_error)
    run_interview(brief, recordings_dir=str(rec), wait_for_enter=enter_log()[1], say=lambda *a: None)
    assert sorted(os.listdir(rec)) == ["notes.txt", "q1.wav"]      # this run's videos deleted, the rest kept
    assert (rec / "q1.wav").read_bytes() == b"old"


def test_a_partial_file_from_a_failing_device_is_deleted(brief, tmp_path, monkeypatch, devices):
    rec = tmp_path / "rec"

    def half_written(path, stop=None, max_seconds=0, camera=None):
        open(path, "wb").write(b"half")                           # the device started writing, then failed
        raise VisionError("webcam unplugged")
    monkeypatch.setattr(interview_vision, "record_video", half_written)
    run_interview(brief, recordings_dir=str(rec), wait_for_enter=enter_log()[1], say=lambda *a: None)
    assert not rec.exists()


def test_a_run_that_cannot_start_deletes_nothing(tmp_path, devices):
    out = tmp_path / "output"
    out.mkdir()
    rec = tmp_path / "rec"
    rec.mkdir()
    (rec / "q1.wav").write_bytes(b"earlier interview")
    with pytest.raises(BriefError):
        run_interview(str(out / "interview_brief.json"), recordings_dir=str(rec))
    assert (rec / "q1.wav").read_bytes() == b"earlier interview"


def test_nothing_recorded_says_so(brief, tmp_path, monkeypatch, devices):
    def mic_error(path, max_seconds=0, stop=None):
        raise SpeechError("no microphone")

    def cam_error(path, stop=None, max_seconds=0, camera=None):
        raise VisionError("no webcam")
    monkeypatch.setattr(interview_speech, "record_audio", mic_error)
    monkeypatch.setattr(interview_vision, "record_video", cam_error)
    said = []
    run_interview(brief, recordings_dir=str(tmp_path / "rec"), wait_for_enter=enter_log()[1], say=said.append)
    assert said[-1] == "No recordings were made."


def test_command_line_keep_recordings_flag(brief, monkeypatch):
    seen = {}

    def fake(brief_path, **kwargs):
        seen.update(kwargs)
        return {"overall": {"questions_answered": 1, "questions_total": 4, "speech": None, "vision": None},
                "questions": []}
    monkeypatch.setattr(interview_agent, "run_interview", fake)
    assert interview_agent.main(["--brief", brief]) == 0
    assert seen["keep_recordings"] is False
    assert interview_agent.main(["--brief", brief, "--keep-recordings"]) == 0
    assert seen["keep_recordings"] is True


# ---------- analyse_and_save: the analysis + saving part of run_interview ----------

def test_analyse_and_save_with_fake_analysers(brief, tmp_path, monkeypatch):
    monkeypatch.setattr(interview_speech, "analyze_audio",
                        lambda path: {"duration_seconds": 2.0, "word_count": 7, "words_per_minute": 150.0,
                                      "filler_count": 1, "fillers": {"um": 1}, "long_pause_count": 0,
                                      "speaking_seconds": 2.0})
    monkeypatch.setattr(interview_vision, "analyze_video", lambda path: dict(VISION))
    rec = tmp_path / "rec"
    rec.mkdir()
    audio, video = rec / "q1.wav", rec / "q1.avi"
    audio.write_bytes(b"a")
    video.write_bytes(b"v")
    loaded = interview_agent.build_interview(brief)
    questions = loaded["questions"]
    answers = [interview_agent.new_answer(questions[0], {"audio": str(audio), "video": str(video), "errors": []}),
               interview_agent.new_answer(questions[1], {"audio": None, "video": None,
                                                         "errors": ["speech: no microphone"]})]
    progress, said = [], []
    results = interview_agent.analyse_and_save(answers, loaded["brief"], brief, "live", str(rec),
                                               keep_recordings=False, say=said.append,
                                               on_answer=lambda n, total: progress.append((n, total)))
    assert progress == [(1, 2), (2, 2)]
    first, second = results["questions"]
    assert first["answered"] and first["speech"]["word_count"] == 7 and first["vision"]["eye_contact_percent"] == 80.0
    assert first["answer_seconds"] == 3.0                               # the longer of speech (2.0) and vision (3.0)
    assert second["answered"] is False and second["errors"] == ["speech: no microphone"]
    assert "_files" not in first and results["overall"]["questions_answered"] == 1
    assert results_file(brief) == json.loads(json.dumps(results))        # saved next to the brief
    assert not rec.exists()                                              # recordings policy applied: deleted
    assert first["recordings"] == {"audio": None, "video": None}
    assert said[-1] == "Recordings deleted."
