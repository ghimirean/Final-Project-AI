"""Tests for Agent 4, step 4a: combining the per-answer metrics into the overall
numbers and saving interview_results.json (agents/interview_results.py). All data is fake."""
import json
import os

import pytest

import interview_results as ir


def speech(words=100, seconds=50.0, fillers=None, pauses=0, duration=60.0, pace="auto"):
    fillers = fillers or {}
    wpm = round(words / (seconds / 60), 1) if pace == "auto" else pace
    return {"transcript": "x", "duration_seconds": duration, "word_count": words, "speaking_seconds": seconds,
            "words_per_minute": wpm, "filler_count": sum(fillers.values()),
            "filler_rate_per_100_words": 0.0, "fillers": fillers, "long_pause_count": pauses,
            "long_pauses": [], "seconds_before_first_word": 1.0}


def vision(frames=100, eye=80.0, upright=90.0, face=100.0, pose=100.0, smile=10.0, look=2.0):
    return {"duration_seconds": frames / 5, "frames_analyzed": frames, "face_detected_percent": face,
            "eye_contact_percent": eye, "longest_look_away_seconds": look, "pose_detected_percent": pose,
            "posture_upright_percent": upright, "average_shoulder_tilt_degrees": 2.0, "smile_percent": smile}


def answer(sp=None, vi=None):
    return {"id": 1, "kind": "intro", "answered": sp is not None or vi is not None, "speech": sp, "vision": vi,
            "errors": [], "recordings": {"audio": None, "video": None}, "question": "q", "topic": None,
            "source": "template", "answer_seconds": 1.0}


def test_overall_speech_totals_and_pace():
    a, b = answer(speech(100, 50.0, {"um": 2, "like": 1}, pauses=1)), answer(speech(60, 30.0, {"um": 1}, pauses=2))
    o = ir.overall_speech([a, b])
    assert o["total_words"] == 160 and o["words_per_minute"] == 120.0      # 160 words / 80 s
    assert o["filler_count"] == 4 and o["fillers"] == {"um": 3, "like": 1}
    assert o["filler_rate_per_100_words"] == 2.5 and o["long_pause_count"] == 3
    assert o["average_answer_seconds"] == 60.0 and o["answers_analyzed"] == 2


def test_overall_pace_ignores_answers_too_short_for_a_pace():
    o = ir.overall_speech([answer(speech(100, 50.0)), answer(speech(3, 2.0, pace=None))])
    assert o["total_words"] == 103 and o["words_per_minute"] == 120.0
    assert ir.overall_speech([answer(speech(3, 2.0, pace=None))])["words_per_minute"] is None


def test_overall_speech_with_no_speech():
    assert ir.overall_speech([answer(None, vision())]) is None
    assert ir.overall_speech([]) is None


def test_overall_vision_is_weighted_by_frames():
    long_answer, short_answer = answer(None, vision(300, eye=90.0, look=4.0)), answer(None, vision(100, eye=50.0, look=9.0))
    o = ir.overall_vision([long_answer, short_answer])
    assert o["eye_contact_percent"] == 80.0                # (90*300 + 50*100) / 400
    assert o["frames_analyzed"] == 400 and o["longest_look_away_seconds"] == 9.0


def test_overall_vision_skips_answers_with_no_posture_or_none_values():
    no_body = vision(100, upright=None, pose=0.0, smile=None, face=0.0, eye=None)
    o = ir.overall_vision([answer(None, no_body), answer(None, vision(100, upright=60.0))])
    assert o["posture_upright_percent"] == 60.0 and o["eye_contact_percent"] == 80.0
    only = ir.overall_vision([answer(None, no_body)])
    assert only["posture_upright_percent"] is None and only["smile_percent"] is None and only["eye_contact_percent"] is None


def test_overall_vision_with_no_video():
    assert ir.overall_vision([answer(speech())]) is None


BRIEF = {"brief_version": 1, "created": "2026-11-18T10:00:00",
         "job": {"id": "j1", "title": "Data Analyst", "company": "Acme"}}


def test_build_results_shape_and_counts():
    unanswered = dict(answer(), answered=False, speech=None, vision=None)
    results = ir.build_results(BRIEF, "/x/output/interview_brief.json",
                               [answer(speech(), vision()), unanswered], "files", {"whisper_model": "base.en"})
    assert results["results_version"] == 1 and results["mode"] == "files"
    assert results["job"] == {"id": "j1", "title": "Data Analyst", "company": "Acme"}
    assert results["brief"] == {"path": "interview_brief.json", "created": "2026-11-18T10:00:00"}
    assert results["overall"]["questions_total"] == 2 and results["overall"]["questions_answered"] == 1
    assert results["overall"]["speech"]["total_words"] == 100
    assert results["settings"] == {"whisper_model": "base.en"}
    json.dumps(results)


def test_nothing_answered_still_builds():
    results = ir.build_results(BRIEF, "b.json", [dict(answer(), answered=False, speech=None, vision=None)], "live", {})
    assert results["overall"] == {"questions_total": 1, "questions_answered": 0, "speech": None, "vision": None}


def test_save_and_remove(tmp_path):
    out = tmp_path / "output"
    path = ir.save_results({"a": "é"}, str(out))
    assert path == str(out / "interview_results.json") and json.load(open(path, encoding="utf-8")) == {"a": "é"}
    assert not os.path.exists(path + ".tmp")
    ir.remove_old_results(str(out))
    assert not os.path.exists(path)
    ir.remove_old_results(str(out))                       # nothing there: no error
