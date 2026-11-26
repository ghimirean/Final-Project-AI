"""Tests for agents/interview_recorder.py (the interview page's start/stop recorder).

The microphone and webcam are FAKES (interview_speech.record_audio and
interview_vision.record_video are replaced): nothing is ever recorded for
real. Files go to a temp folder."""
import os
import re
import threading
import time

import pytest

import interview_recorder
import interview_speech
import interview_vision
from interview_agent import MAX_ANSWER_SECONDS
from interview_recorder import AnswerRecorder, new_recordings_dir
from interview_speech import SpeechError
from interview_vision import VisionError


def wait_for(stop, limit=3.0):
    end = time.time() + limit
    while not stop() and time.time() < end:
        time.sleep(0.01)


@pytest.fixture
def devices(monkeypatch):
    """Fake microphone and webcam: record until stopped, then write a small file.
    Returns a log of (device, file name, max_seconds)."""
    log = []

    def record_audio(path, max_seconds=0, stop=None):
        log.append(("audio", os.path.basename(path), max_seconds))
        wait_for(stop)
        open(path, "wb").write(b"fake audio")

    def record_video(path, stop=None, max_seconds=0, camera=None):
        log.append(("video", os.path.basename(path), max_seconds))
        wait_for(stop)
        open(path, "wb").write(b"fake video")
    monkeypatch.setattr(interview_speech, "record_audio", record_audio)
    monkeypatch.setattr(interview_vision, "record_video", record_video)
    return log


def test_both_devices_start_together(tmp_path, monkeypatch):
    barrier = threading.Barrier(2, timeout=3)       # passes only if both are recording at the same time

    def audio(path, max_seconds=0, stop=None):
        barrier.wait()
        wait_for(stop)
        open(path, "wb").write(b"a")

    def video(path, stop=None, max_seconds=0, camera=None):
        barrier.wait()
        wait_for(stop)
        open(path, "wb").write(b"v")
    monkeypatch.setattr(interview_speech, "record_audio", audio)
    monkeypatch.setattr(interview_vision, "record_video", video)
    recorder = AnswerRecorder(1, str(tmp_path / "rec"))
    recorder.start()
    got = recorder.stop()
    assert got["errors"] == [] and got["audio"] and got["video"]


def test_start_returns_at_once_and_stop_returns_the_paths(tmp_path, devices):
    rec = tmp_path / "rec"
    recorder = AnswerRecorder(3, str(rec))
    assert not recorder.started and not recorder.recording
    started = time.time()
    recorder.start()
    assert time.time() - started < 1.0                     # doesn't wait for the answer
    assert recorder.started and recorder.recording
    got = recorder.stop()
    assert got == {"audio": str(rec / "q3.wav"), "video": str(rec / "q3.avi"), "errors": []}
    assert (rec / "q3.wav").read_bytes() == b"fake audio" and (rec / "q3.avi").read_bytes() == b"fake video"
    assert not recorder.recording


def test_files_are_named_per_question_in_the_given_folder(tmp_path, devices):
    rec = tmp_path / "rec" / "20261125-120000"
    for number in (1, 2):
        recorder = AnswerRecorder(number, str(rec))
        recorder.start()
        recorder.stop()
    assert sorted(os.listdir(rec)) == ["q1.avi", "q1.wav", "q2.avi", "q2.wav"]
    assert {name for _, name, _ in devices} == {"q1.wav", "q1.avi", "q2.wav", "q2.avi"}


def test_new_recordings_dir_is_a_date_time_folder(tmp_path):
    folder = new_recordings_dir(str(tmp_path))
    assert os.path.dirname(folder) == str(tmp_path)
    assert re.fullmatch(r"\d{8}-\d{6}", os.path.basename(folder))
    assert not os.path.exists(folder)                      # created only when recording starts


def test_the_120_second_limit_is_passed_to_both_devices(tmp_path, devices):
    recorder = AnswerRecorder(1, str(tmp_path / "rec"))
    recorder.start()
    recorder.stop()
    assert MAX_ANSWER_SECONDS == 120
    assert sorted((d, s) for d, _, s in devices) == [("audio", 120), ("video", 120)]


def test_a_custom_limit_is_passed_too(tmp_path, devices):
    recorder = AnswerRecorder(1, str(tmp_path / "rec"), max_seconds=45)
    recorder.start()
    recorder.stop()
    assert {s for _, _, s in devices} == {45}


def test_recording_ends_by_itself_at_the_limit(tmp_path, monkeypatch):
    def short(path, max_seconds=0, stop=None, camera=None):
        open(path, "wb").write(b"x")                       # returns as if the time limit was reached
    monkeypatch.setattr(interview_speech, "record_audio", short)
    monkeypatch.setattr(interview_vision, "record_video", short)
    recorder = AnswerRecorder(1, str(tmp_path / "rec"), max_seconds=1)
    recorder.start()
    end = time.time() + 3
    while recorder.recording and time.time() < end:
        time.sleep(0.01)
    assert not recorder.recording                          # the page then asks the user to click Stop
    assert recorder.stop()["audio"] and recorder.result["video"]


def test_a_failing_webcam_does_not_stop_the_microphone(tmp_path, monkeypatch, devices):
    def no_camera(path, stop=None, max_seconds=0, camera=None):
        raise VisionError("Could not open the webcam.")
    monkeypatch.setattr(interview_vision, "record_video", no_camera)
    recorder = AnswerRecorder(2, str(tmp_path / "rec"))
    recorder.start()
    time.sleep(0.05)
    assert recorder.recording                              # the microphone is still recording
    got = recorder.stop()
    assert got["audio"].endswith("q2.wav") and got["video"] is None
    assert got["errors"] == ["vision: Could not open the webcam."]


def test_a_failing_microphone_does_not_stop_the_webcam(tmp_path, monkeypatch, devices):
    def no_mic(path, max_seconds=0, stop=None):
        raise SpeechError("no microphone")
    monkeypatch.setattr(interview_speech, "record_audio", no_mic)
    recorder = AnswerRecorder(1, str(tmp_path / "rec"))
    recorder.start()
    got = recorder.stop()
    assert got["audio"] is None and got["video"].endswith("q1.avi")
    assert got["errors"] == ["speech: no microphone"]


def test_an_unexpected_device_crash_is_an_error_not_a_crash(tmp_path, monkeypatch, devices):
    def crash(path, stop=None, max_seconds=0, camera=None):
        raise RuntimeError("driver exploded")
    monkeypatch.setattr(interview_vision, "record_video", crash)
    recorder = AnswerRecorder(1, str(tmp_path / "rec"))
    recorder.start()
    assert recorder.stop()["errors"] == ["vision: unexpected recording error: driver exploded"]


def test_stop_twice_gives_the_same_result_and_start_twice_is_refused(tmp_path, devices):
    recorder = AnswerRecorder(1, str(tmp_path / "rec"))
    recorder.start()
    with pytest.raises(RuntimeError):
        recorder.start()
    first = recorder.stop()
    assert recorder.stop() is first


# ---------- created_files: the only files the delete policy may touch ----------

def test_created_files_are_the_recordings(tmp_path, devices):
    rec = tmp_path / "rec"
    recorder = AnswerRecorder(1, str(rec))
    recorder.start()
    recorder.stop()
    assert recorder.created_files() == [str(rec / "q1.wav"), str(rec / "q1.avi")]


def test_a_file_that_was_already_there_is_not_created(tmp_path, monkeypatch, devices):
    rec = tmp_path / "rec"
    rec.mkdir()
    (rec / "q1.wav").write_bytes(b"older interview")      # this time the microphone fails

    def no_mic(path, max_seconds=0, stop=None):
        raise SpeechError("no microphone")
    monkeypatch.setattr(interview_speech, "record_audio", no_mic)
    recorder = AnswerRecorder(1, str(rec))
    recorder.start()
    recorder.stop()
    assert recorder.created_files() == [str(rec / "q1.avi")]


def test_a_partial_file_from_a_failing_device_is_created(tmp_path, monkeypatch, devices):
    rec = tmp_path / "rec"

    def half_written(path, stop=None, max_seconds=0, camera=None):
        open(path, "wb").write(b"half")
        raise VisionError("webcam unplugged")
    monkeypatch.setattr(interview_vision, "record_video", half_written)
    recorder = AnswerRecorder(1, str(rec))
    recorder.start()
    got = recorder.stop()
    assert got["video"] is None
    assert recorder.created_files() == [str(rec / "q1.wav"), str(rec / "q1.avi")]


def test_nothing_created_before_start(tmp_path):
    assert AnswerRecorder(1, str(tmp_path / "rec")).created_files() == []
    assert interview_recorder.RECORDINGS_DIR.endswith("recordings")
