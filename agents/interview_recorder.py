"""Records one interview answer in the background, for a UI that starts and
stops it with buttons (interview_ui.py).

It mirrors interview_agent.record_answer(): the microphone
(interview_speech.record_audio) and the webcam (interview_vision.record_video)
record at the same time in two threads, each with a stop callback and the
120 s limit, into q{n}.wav / q{n}.avi in this interview's recordings folder.
One failing device never stops the other. The difference: record_answer waits
for Enter in the terminal, while here start() returns at once and stop() is
called later (by a "Stop" button).

    recorder = AnswerRecorder(2, new_recordings_dir())
    recorder.start()
    ...
    got = recorder.stop()      # {"audio": path or None, "video": path or None, "errors": [...]}
"""
import os
import threading
from datetime import datetime

import interview_speech
import interview_vision
from interview_agent import MAX_ANSWER_SECONDS, RECORDINGS_DIR
from interview_speech import SpeechError
from interview_vision import VisionError


def new_recordings_dir(base=RECORDINGS_DIR):
    """This interview's folder: recordings/<date-time>/, like Agent 4's."""
    return os.path.join(base, datetime.now().strftime("%Y%m%d-%H%M%S"))


class AnswerRecorder:
    """Microphone + webcam for one answer, in two background threads."""

    def __init__(self, number, recordings_dir, max_seconds=MAX_ANSWER_SECONDS):
        self.number, self.recordings_dir, self.max_seconds = number, recordings_dir, max_seconds
        self.audio_path = os.path.join(recordings_dir, f"q{number}.wav")
        self.video_path = os.path.join(recordings_dir, f"q{number}.avi")
        self._stop = threading.Event()
        self._threads = []
        self._lock = threading.Lock()
        self._out = {"audio": None, "video": None, "errors": []}
        self.result = None                      # the stop() result, once stopped
        self._existed = set()

    def _run(self, kind, path, error_type):
        # Looked up when the thread runs, so tests can replace the recorders with fakes
        recorder = interview_speech.record_audio if kind == "audio" else interview_vision.record_video
        name = "speech" if kind == "audio" else "vision"
        try:
            recorder(path, max_seconds=self.max_seconds, stop=self._stop.is_set)
            with self._lock:
                self._out[kind] = path
        except error_type as e:
            with self._lock:
                self._out["errors"].append(f"{name}: {e}")
        except Exception as e:                               # a device driver failing in an odd way
            with self._lock:
                self._out["errors"].append(f"{name}: unexpected recording error: {e}")

    @property
    def started(self):
        return bool(self._threads)

    @property
    def recording(self):
        """True between start() and stop() while at least one device is still recording
        (False once both have finished: stopped, failed, or hit the time limit)."""
        return self.result is None and any(t.is_alive() for t in self._threads)

    def start(self):
        """Starts both devices and returns at once. Can be called only once."""
        if self._threads:
            raise RuntimeError("this answer is already being recorded")
        os.makedirs(self.recordings_dir, exist_ok=True)
        self._existed = {p for p in (self.audio_path, self.video_path) if os.path.exists(p)}
        self._threads = [
            threading.Thread(target=self._run, args=("audio", self.audio_path, SpeechError), daemon=True),
            threading.Thread(target=self._run, args=("video", self.video_path, VisionError), daemon=True)]
        for t in self._threads:
            t.start()

    def stop(self):
        """Stops both devices, waits for them, and returns {"audio", "video", "errors"}
        (paths are None for a device that recorded nothing). Calling it again
        returns the same result."""
        if self.result is not None:
            return self.result
        self._stop.set()
        for t in self._threads:
            t.join()
        with self._lock:
            self.result = {"audio": self._out["audio"], "video": self._out["video"],
                           "errors": list(self._out["errors"])}
        return self.result

    def created_files(self):
        """The files this recording made (after stop()): a finished recording, or a
        partial file a failing device left behind. A file that was already there
        before start() and wasn't recorded now is never included (same rule as
        run_interview), so the delete-after-analysis policy can't touch it."""
        got = self.result or {}
        return [p for p in (self.audio_path, self.video_path)
                if p in (got.get("audio"), got.get("video"))
                or (p not in self._existed and os.path.exists(p))]
