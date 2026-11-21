"""Agent 4, step 4: build and save output/interview_results.json.

The file is the contract with the scorecard / feedback page (README, "Agent 4").
Everything here is plain Python on the per-answer metrics from
interview_speech.py and interview_vision.py: no models, no hardware.
"""
import json
import os
from datetime import datetime

RESULTS_VERSION = 1
RESULTS_FILENAME = "interview_results.json"


def _weighted_mean(pairs):
    """[(value, weight)] -> weighted mean rounded to 1 decimal, ignoring None values; None if nothing is left."""
    pairs = [(v, w) for v, w in pairs if v is not None and w]
    total = sum(w for _, w in pairs)
    return round(sum(v * w for v, w in pairs) / total, 1) if total else None


def overall_speech(answers):
    """Speech metrics over all answers. words_per_minute = all words / all speaking time
    (answers with too few words for a pace are left out of the time)."""
    speech = [a["speech"] for a in answers if a.get("speech")]
    if not speech:
        return None
    words = sum(s["word_count"] for s in speech)
    paced = [s for s in speech if s["words_per_minute"] is not None]
    paced_words, paced_seconds = sum(s["word_count"] for s in paced), sum(s["speaking_seconds"] for s in paced)
    fillers = {}
    for s in speech:
        for word, count in s["fillers"].items():
            fillers[word] = fillers.get(word, 0) + count
    filler_count = sum(s["filler_count"] for s in speech)
    return {
        "answers_analyzed": len(speech),
        "total_words": words,
        "words_per_minute": round(paced_words / (paced_seconds / 60.0), 1) if paced_seconds > 0 else None,
        "filler_count": filler_count,
        "filler_rate_per_100_words": round(100 * filler_count / words, 1) if words else 0.0,
        "fillers": dict(sorted(fillers.items(), key=lambda kv: (-kv[1], kv[0]))),
        "long_pause_count": sum(s["long_pause_count"] for s in speech),
        "average_answer_seconds": round(sum(s["duration_seconds"] for s in speech) / len(speech), 1),
    }


def overall_vision(answers):
    """Vision metrics over all answers, each answer weighted by the frames analysed
    (a longer answer counts for more)."""
    vision = [a["vision"] for a in answers if a.get("vision")]
    if not vision:
        return None
    frames = lambda v: v["frames_analyzed"]       # noqa: E731
    looks = [v["longest_look_away_seconds"] for v in vision if v["longest_look_away_seconds"] is not None]
    return {
        "answers_analyzed": len(vision),
        "frames_analyzed": sum(frames(v) for v in vision),
        "face_detected_percent": _weighted_mean([(v["face_detected_percent"], frames(v)) for v in vision]),
        "eye_contact_percent": _weighted_mean([(v["eye_contact_percent"], frames(v)) for v in vision]),
        "longest_look_away_seconds": max(looks) if looks else None,
        "posture_upright_percent": _weighted_mean([(v["posture_upright_percent"], frames(v) * (v["pose_detected_percent"] or 0))
                                                   for v in vision]),
        "smile_percent": _weighted_mean([(v["smile_percent"], frames(v) * (v["face_detected_percent"] or 0))
                                         for v in vision]),
    }


def build_results(brief, brief_path, answers, mode, settings):
    """The whole results document. `answers` has one dict per question:
    the question fields + answered, speech, vision, errors, recordings."""
    answered = [a for a in answers if a["answered"]]
    return {
        "results_version": RESULTS_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,                                             # "live" or "files"
        "brief": {"path": os.path.basename(brief_path), "created": brief.get("created")},
        "job": {"id": brief["job"].get("id"), "title": brief["job"].get("title"),
                "company": brief["job"].get("company")},
        "questions": answers,
        "overall": {"questions_total": len(answers), "questions_answered": len(answered),
                    "speech": overall_speech(answered), "vision": overall_vision(answered)},
        "settings": settings,
    }


def save_results(results, out_dir):
    """Writes out_dir/interview_results.json (all at once, so a crash can't leave half a file). Returns the path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, RESULTS_FILENAME)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.replace(path + ".tmp", path)
    return path


def remove_old_results(out_dir):
    """Deletes a previous interview_results.json, so the file exists only if THIS interview wrote it."""
    try:
        os.remove(os.path.join(out_dir, RESULTS_FILENAME))
    except FileNotFoundError:
        pass
