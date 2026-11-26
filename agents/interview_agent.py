"""Agent 4: the mock interview (computer vision + speech).

Entry point: run_interview(brief_path). The orchestrator calls it after the
user picks a job, i.e. after it prints "Interview brief for Agent 4".

    python agents/interview_agent.py                         # live: microphone + webcam
    python agents/interview_agent.py --questions-only        # just print the questions
    python agents/interview_agent.py --audio a1.wav a2.wav --video v1.avi v2.avi
                                                             # pre-recorded answers, in question order
    python agents/interview_agent.py --keep-recordings       # live, but keep the audio + video afterwards

Flow: read the brief -> build 4-5 template questions (interview_questions.py)
-> for each question record the answer (microphone + webcam at the same time)
-> Whisper + MediaPipe measure each answer (interview_speech.py, interview_vision.py)
-> write output/interview_results.json (interview_results.py).
Recording comes first and analysis after, so there is no waiting between questions.
"""
import argparse
import os
import sys
import threading
import time
from datetime import datetime

import interview_speech
import interview_vision
from interview_questions import BriefError, build_questions, load_brief
from interview_results import build_results, remove_old_results, save_results
from interview_speech import SpeechError
from interview_vision import VisionError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "output")
RECORDINGS_DIR = os.path.join(ROOT, "recordings")          # git-ignored
DEFAULT_BRIEF = os.path.join(OUTPUT_DIR, "interview_brief.json")
MAX_ANSWER_SECONDS = 120
DEVICE_START_GRACE = 0.3        # seconds to see whether the microphone and webcam started at all


class InterviewError(Exception):
    """The interview can't run as asked (for example more answer files than questions)."""


def build_interview(brief_path=DEFAULT_BRIEF):
    """Reads the brief and builds the questions only. Raises BriefError when there is
    no usable brief (so Agent 4 never starts from a missing or stale one)."""
    brief = load_brief(brief_path)
    job = brief["job"]
    return {"brief": brief,
            "job": {"id": job.get("id"), "title": job.get("title"), "company": job.get("company")},
            "questions": build_questions(brief)}


# ---------- recording one answer ----------

def _rel(path):
    """A path for the results file: relative to the project when it is inside it."""
    if not path:
        return None
    path = os.path.abspath(path)
    return os.path.relpath(path, ROOT) if path.startswith(ROOT + os.sep) else path


def record_answer(number, recordings_dir, wait_for_enter, say, max_seconds=MAX_ANSWER_SECONDS):
    """Records one answer: microphone and webcam at the same time, until the user presses
    Enter (or max_seconds pass, then Enter just continues). One failing device doesn't
    stop the other. Returns {"audio", "video", "errors"} with file paths (None if not recorded)."""
    stop = threading.Event()
    out = {"audio": None, "video": None, "errors": []}
    audio_path = os.path.join(recordings_dir, f"q{number}.wav")
    video_path = os.path.join(recordings_dir, f"q{number}.avi")

    def run(kind, path, recorder, error_type):
        try:
            recorder(path, max_seconds=max_seconds, stop=stop.is_set)
            out[kind] = path
        except error_type as e:
            out["errors"].append(f"{'speech' if kind == 'audio' else 'vision'}: {e}")
        except Exception as e:                                   # a device driver failing in an odd way
            out["errors"].append(f"{'speech' if kind == 'audio' else 'vision'}: unexpected recording error: {e}")

    threads = [threading.Thread(target=run, args=("audio", audio_path, interview_speech.record_audio, SpeechError)),
               threading.Thread(target=run, args=("video", video_path, interview_vision.record_video, VisionError))]
    os.makedirs(recordings_dir, exist_ok=True)
    wait_for_enter("Press Enter to start your answer... ")
    say(f"Recording (up to {max_seconds} s). Look at the camera and answer out loud. Press Enter when you are done.")
    for t in threads:
        t.start()
    time.sleep(DEVICE_START_GRACE)
    if not any(t.is_alive() for t in threads) and len(out["errors"]) == 2:
        say("Nothing could be recorded for this question.")        # both devices failed at once: don't ask for Enter
    else:
        wait_for_enter("")
        stop.set()
    for t in threads:
        t.join()
    return out


# ---------- the interview ----------

def delete_recordings(created, recordings_dir):
    """Deletes the files THIS run recorded (`created`), then this run's recordings
    folder if it is empty. Only paths inside recordings_dir are touched. Returns
    (deleted paths, paths that could not be deleted)."""
    folder = os.path.abspath(recordings_dir)
    deleted, failed = [], []
    for path in created:
        path = os.path.abspath(path)
        if os.path.dirname(path) != folder:                  # never anything outside this run's folder
            continue
        try:
            os.remove(path)
            deleted.append(path)
        except FileNotFoundError:
            deleted.append(path)
        except OSError:
            failed.append(path)
    try:
        if os.path.isdir(folder) and not os.listdir(folder):
            os.rmdir(folder)
    except OSError:
        pass
    return deleted, failed


def run_interview(brief_path=DEFAULT_BRIEF, audio_files=None, video_files=None, recordings_dir=None,
                  wait_for_enter=input, say=print, max_answer_seconds=MAX_ANSWER_SECONDS,
                  keep_recordings=False):
    """Runs the mock interview and writes interview_results.json next to the brief
    (output/interview_results.json). Returns the same results as a dict.

    Live by default (microphone + webcam). Give `audio_files` and/or `video_files`
    (lists in question order, None to skip one) to analyse pre-recorded answers instead.
    Raises BriefError if there is no usable brief; a failing microphone, webcam or
    model is written into the results (per question "errors") instead of stopping the interview.

    Live mode: once the answers are analysed and the results saved, the audio and
    video files recorded in this run are deleted (and the run's recordings folder if
    it is then empty), and their "recordings" paths in the results become null.
    keep_recordings=True keeps them. Pre-recorded files (mode "files") are never deleted."""
    out_dir = os.path.dirname(os.path.abspath(brief_path))
    remove_old_results(out_dir)         # first of all: a failed or half-finished run must not leave old results behind
    interview = build_interview(brief_path)
    brief, questions = interview["brief"], interview["questions"]
    mode = "files" if (audio_files or video_files) else "live"
    audio_files, video_files = list(audio_files or []), list(video_files or [])
    if max(len(audio_files), len(video_files)) > len(questions):
        raise InterviewError(f"This interview has {len(questions)} questions, but "
                             f"{max(len(audio_files), len(video_files))} answer files were given.")
    answers = []
    created = []                        # files THIS run recorded (live mode only): the only ones ever deleted
    if mode == "live":
        recordings_dir = recordings_dir or os.path.join(RECORDINGS_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))
        job = interview["job"]
        say(f"\nMock interview for {job['title']}" + (f" at {job['company']}" if job["company"] else "")
            + f": {len(questions)} questions. Your answers are recorded on this computer only.")
    for index, q in enumerate(questions):
        if mode == "live":
            say(f"\nQuestion {q['id']} of {len(questions)}: {q['question']}")
            planned = [os.path.join(recordings_dir, f"q{q['id']}.{ext}") for ext in ("wav", "avi")]
            existed = {p for p in planned if os.path.exists(p)}
            got = record_answer(q["id"], recordings_dir, wait_for_enter, say, max_answer_seconds)
            # recorded now: a finished recording, or a partial file a failing device left behind
            created += [p for p in planned
                        if p in (got["audio"], got["video"]) or (p not in existed and os.path.exists(p))]
        else:
            got = {"audio": audio_files[index] if index < len(audio_files) else None,
                   "video": video_files[index] if index < len(video_files) else None, "errors": []}
        answers.append(new_answer(q, got))

    return analyse_and_save(answers, brief, brief_path, mode, recordings_dir, keep_recordings,
                            created=created, say=say)


def new_answer(question, got):
    """One answer before analysis: the question's fields + the recorded or given
    files (`got` = {"audio", "video", "errors"}, as record_answer returns)."""
    return dict(question, answered=False, answer_seconds=None, speech=None, vision=None,
                errors=list(got["errors"]),
                recordings={"audio": _rel(got["audio"]), "video": _rel(got["video"])},
                _files=got)


def analyse_and_save(answers, brief, brief_path, mode, recordings_dir=None, keep_recordings=False,
                     created=None, say=print, on_answer=None):
    """Analyses the answers (new_answer() dicts, in question order) with Whisper and
    MediaPipe, builds the results, saves interview_results.json next to the brief and
    applies the recordings policy: in live mode the files recorded in this run are
    deleted after analysis (their paths become null) unless keep_recordings.
    Returns the results dict.

    created: the files this run recorded, the only ones that are ever deleted
    (default: the answers' own recorded files). on_answer(number, total) is
    called before each answer is analysed (for a progress display)."""
    out_dir = os.path.dirname(os.path.abspath(brief_path))
    if created is None:
        created = [p for a in answers for p in (a["_files"]["audio"], a["_files"]["video"]) if p] \
            if mode == "live" else []
    say("\nAnalysing your answers...")
    for number, answer in enumerate(answers, start=1):
        if on_answer:
            on_answer(number, len(answers))
        files = answer.pop("_files")
        if files["audio"]:
            try:
                answer["speech"] = interview_speech.analyze_audio(files["audio"])
            except SpeechError as e:
                answer["errors"].append(f"speech: {e}")
        if files["video"]:
            try:
                answer["vision"] = interview_vision.analyze_video(files["video"])
            except VisionError as e:
                answer["errors"].append(f"vision: {e}")
        answer["answered"] = answer["speech"] is not None or answer["vision"] is not None
        durations = [m["duration_seconds"] for m in (answer["speech"], answer["vision"]) if m]
        answer["answer_seconds"] = max(durations) if durations else None

    settings = {"whisper_model": interview_speech.SPEECH_CONFIG["model"],
                "long_pause_seconds": interview_speech.LONG_PAUSE_SECONDS,
                "video_analysis_fps": interview_vision.ANALYSIS_FPS}
    results = build_results(brief, brief_path, answers, mode, settings)
    path = save_results(results, out_dir)
    say(f"Interview results saved to {_rel(path)}")
    if mode == "live":
        if keep_recordings:
            say(f"Recordings kept in {_rel(recordings_dir)}" if created else "No recordings were made.")
        elif created:
            deleted, failed = delete_recordings(created, recordings_dir)
            gone = {_rel(p) for p in deleted}
            for answer in results["questions"]:
                answer["recordings"] = {kind: (None if p in gone else p)
                                        for kind, p in answer["recordings"].items()}
            save_results(results, out_dir)                    # the deleted files' paths are now null
            if failed:
                say(f"Recordings deleted, except {len(failed)} file(s) that could not be deleted in "
                    f"{_rel(recordings_dir)}")
            else:
                say("Recordings deleted.")
        else:
            say("No recordings were made.")
    return results


def print_summary(results, say=print):
    overall = results["overall"]
    say(f"\nAnswered {overall['questions_answered']} of {overall['questions_total']} questions.")
    speech, vision = overall["speech"], overall["vision"]
    if speech:
        pace = f"{speech['words_per_minute']} words/min" if speech["words_per_minute"] is not None else "pace n/a"
        say(f"Speech: {speech['total_words']} words, {pace}, {speech['filler_count']} filler words, "
            f"{speech['long_pause_count']} long pauses")
    if vision:
        say(f"Vision: eye contact {vision['eye_contact_percent']}%, upright posture {vision['posture_upright_percent']}%")
    for q in results["questions"]:
        for error in q["errors"]:
            say(f"Question {q['id']}: {error}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="INTERVAI Agent 4: mock interview (speech + vision)")
    parser.add_argument("--brief", default=DEFAULT_BRIEF, help="interview brief (default output/interview_brief.json)")
    parser.add_argument("--questions-only", action="store_true", help="print the questions and stop")
    parser.add_argument("--audio", nargs="+", metavar="FILE", help="pre-recorded answers (audio), in question order")
    parser.add_argument("--video", nargs="+", metavar="FILE", help="pre-recorded answers (video), in question order")
    parser.add_argument("--keep-recordings", action="store_true",
                        help="keep this interview's audio and video in recordings/ (deleted after analysis by default)")
    args = parser.parse_args(argv)
    try:
        if args.questions_only:
            interview = build_interview(args.brief)
            job = interview["job"]
            print(f"Interview for {job['title']}" + (f" at {job['company']}" if job["company"] else ""))
            for q in interview["questions"]:
                print(f"\nQ{q['id']} [{q['kind']}] {q['question']}")
            return 0
        results = run_interview(args.brief, audio_files=args.audio, video_files=args.video,
                                keep_recordings=args.keep_recordings)
    except (BriefError, InterviewError) as e:
        print(f"Agent 4 cannot start: {e}", file=sys.stderr)
        return 1
    print_summary(results)
    return 0 if results["overall"]["questions_answered"] else 1      # nothing could be recorded or analysed


if __name__ == "__main__":
    sys.exit(main())
