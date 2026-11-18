"""Tests for Agent 4, step 3: vision (agents/interview_vision.py).
Offline: landmarks are made-up numbers, the camera and MediaPipe are fakes
(the real ones are blocked in tests/conftest.py), videos are tiny synthetic files."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

import interview_vision as iv
from interview_vision import VisionError


# ---------- fake landmarks ----------

def pt(x, y, visibility=1.0):
    return SimpleNamespace(x=x, y=y, visibility=visibility)


def face(yaw=0.5, pitch=0.62):
    """478 landmarks. The face is 0.2 wide and 0.4 tall; yaw/pitch place the nose in it
    (0.5 = middle of the width, pitch 0 = forehead, 1 = chin)."""
    points = [pt(0.5, 0.5) for _ in range(478)]
    points[iv.FACE_EDGE_A] = pt(0.4, 0.5)
    points[iv.FACE_EDGE_B] = pt(0.6, 0.5)
    points[iv.FOREHEAD] = pt(0.5, 0.3)
    points[iv.CHIN] = pt(0.5, 0.7)
    points[iv.NOSE] = pt(0.4 + 0.2 * yaw, 0.3 + 0.4 * pitch)
    return points


def body(neck=0.45, tilt_dy=0.0, nose_dx=0.0, visibility=1.0):
    """33 pose landmarks. Shoulders 0.2 apart at y=0.6 (the right one tilt_dy lower);
    the ears sit `neck` shoulder-widths above the shoulder line."""
    points = [pt(0.5, 0.4, visibility) for _ in range(33)]
    points[11], points[12] = pt(0.4, 0.6, visibility), pt(0.6, 0.6 + tilt_dy, visibility)
    ear_y = 0.6 - neck * 0.2
    points[7], points[8] = pt(0.45, ear_y, visibility), pt(0.55, ear_y, visibility)
    points[0] = pt(0.5 + nose_dx, ear_y, visibility)
    return points


# ---------- eye contact and smile ----------

def test_a_centred_face_with_calm_eyes_is_eye_contact():
    assert iv.judge_face(face(), {"eyeLookOutLeft": 0.05}) == {"eye_contact": True, "smile": False}


def test_head_turned_sideways_is_not_eye_contact():
    assert iv.judge_face(face(yaw=0.2), {})["eye_contact"] is False
    assert iv.judge_face(face(yaw=0.8), {})["eye_contact"] is False
    assert iv.judge_face(face(yaw=0.6), {})["eye_contact"] is True       # a small turn is still fine


def test_head_tilted_far_up_or_down_is_not_eye_contact():
    assert iv.judge_face(face(pitch=0.2), {})["eye_contact"] is False     # looking up
    assert iv.judge_face(face(pitch=0.9), {})["eye_contact"] is False     # looking down at notes or a phone


def test_eyes_turned_away_with_the_head_straight():
    for name in ("eyeLookOutLeft", "eyeLookInRight", "eyeLookUpLeft", "eyeLookDownRight"):
        assert iv.judge_face(face(), {name: 0.8})["eye_contact"] is False, name


def test_smile_uses_both_corners_of_the_mouth():
    assert iv.judge_face(face(), {"mouthSmileLeft": 0.7, "mouthSmileRight": 0.6})["smile"] is True
    assert iv.judge_face(face(), {"mouthSmileLeft": 0.7})["smile"] is False        # one side only: 0.35 average


def test_broken_face_geometry_is_not_eye_contact():
    flat = face()
    flat[iv.CHIN] = pt(0.5, 0.3)
    assert iv.judge_face(flat, {}) == {"eye_contact": False, "smile": False}


# ---------- posture ----------

def test_sitting_upright():
    assert iv.judge_posture(body()) == {"upright": True, "shoulder_tilt_degrees": 0.0}


def test_hunched_with_the_head_sunk_between_the_shoulders():
    assert iv.judge_posture(body(neck=0.1))["upright"] is False


def test_one_shoulder_much_lower_is_leaning():
    result = iv.judge_posture(body(tilt_dy=0.08))                 # atan(0.08 / 0.2) = 21.8 degrees
    assert result["upright"] is False and result["shoulder_tilt_degrees"] == pytest.approx(21.8, abs=0.1)
    assert iv.judge_posture(body(tilt_dy=0.02))["upright"] is True    # 5.7 degrees is fine


def test_head_far_to_one_side_of_the_shoulders():
    assert iv.judge_posture(body(nose_dx=0.1))["upright"] is False        # half a shoulder width
    assert iv.judge_posture(body(nose_dx=0.03))["upright"] is True


def test_posture_unknown_when_the_body_is_not_visible():
    assert iv.judge_posture(body(visibility=0.2)) is None
    cropped = body()
    cropped[12] = pt(0.6, 0.6, None)                                # no visibility value at all
    assert iv.judge_posture(cropped) is None


def test_posture_ignores_which_way_round_the_shoulders_are():
    mirrored = body()
    mirrored[11], mirrored[12] = mirrored[12], mirrored[11]          # left/right swapped: the tilt must not become ~180
    assert iv.judge_posture(mirrored)["shoulder_tilt_degrees"] == 0.0


# ---------- one frame ----------

def test_frame_with_no_person():
    o = iv.frame_observation(1.234)
    assert o == {"t": 1.23, "face_found": False, "eye_contact": False, "smile": None,
                 "pose_found": False, "upright": None, "shoulder_tilt_degrees": None}


def test_frame_with_a_person():
    o = iv.frame_observation(0.0, face(), {}, body())
    assert o["face_found"] and o["eye_contact"] and o["pose_found"] and o["upright"]


# ---------- the summary ----------

def obs(eye, upright=True, face_found=True, pose_found=True, smile=False, tilt=2.0):
    return {"t": 0, "face_found": face_found, "eye_contact": eye, "smile": smile if face_found else None,
            "pose_found": pose_found, "upright": upright if pose_found else None,
            "shoulder_tilt_degrees": tilt if pose_found else None}


def test_percentages():
    frames = [obs(True), obs(True), obs(False), obs(False, face_found=False, pose_found=False)]
    s = iv.summarize_frames(frames, 4.0)
    assert s["frames_analyzed"] == 4 and s["duration_seconds"] == 4.0
    assert s["eye_contact_percent"] == 50.0            # a frame with no face counts against eye contact
    assert s["face_detected_percent"] == 75.0 and s["pose_detected_percent"] == 75.0
    assert s["posture_upright_percent"] == 100.0       # only frames where the body was seen
    assert s["average_shoulder_tilt_degrees"] == 2.0


def test_longest_look_away():
    frames = [obs(True), obs(False), obs(False), obs(False), obs(True), obs(False)]
    assert iv.summarize_frames(frames, 6.0)["longest_look_away_seconds"] == 3.0


def test_slouching_and_smiling_percentages():
    frames = [obs(True, upright=False, smile=True), obs(True, upright=True), obs(True, upright=False), obs(True, upright=False)]
    s = iv.summarize_frames(frames, 4.0)
    assert s["posture_upright_percent"] == 25.0 and s["smile_percent"] == 25.0


def test_too_few_frames_gives_none_not_a_wrong_number():
    s = iv.summarize_frames([obs(True), obs(True)], 2.0)
    assert s["frames_analyzed"] == 2 and s["eye_contact_percent"] is None and s["posture_upright_percent"] is None
    empty = iv.summarize_frames([], 0)
    assert empty["frames_analyzed"] == 0 and empty["eye_contact_percent"] is None


def test_body_never_seen_gives_no_posture_number():
    s = iv.summarize_frames([obs(True, pose_found=False) for _ in range(5)], 5.0)
    assert s["pose_detected_percent"] == 0.0 and s["posture_upright_percent"] is None
    assert s["average_shoulder_tilt_degrees"] is None


def test_summary_is_json_serialisable():
    json.dumps(iv.summarize_frames([obs(True)] * 5, 5.0))


# ---------- recording (fake camera, real video writer) ----------

class FakeCamera:
    def __init__(self, frames=None):
        self.frames, self.released, self.reads = frames, False, 0

    def read(self):
        self.reads += 1
        if self.frames is None:
            return True, np.full((48, 64, 3), 120, np.uint8)
        return False, None

    def release(self):
        self.released = True


def test_record_video_writes_a_file_with_real_time_length(tmp_path):
    camera = FakeCamera()
    path = str(tmp_path / "rec" / "q1.avi")
    started = {"n": 0}

    def stop():                       # stop after about 1 second of wall-clock time
        started["n"] += 1
        return started["n"] > 1 and os.path.exists(path) and iv.time.time() - t0 > 1.0
    t0 = iv.time.time()
    seconds = iv.record_video(path, stop=stop, camera=camera)
    assert camera.released
    assert 0.9 <= seconds <= 1.5
    assert abs(iv.video_duration(path) - seconds) < 0.2          # the file's length is the clock's length


def test_record_video_with_no_webcam(tmp_path):
    with pytest.raises(VisionError, match="webcam"):
        iv.record_video(str(tmp_path / "q.avi"), stop=lambda: True)


def test_record_video_that_got_no_picture(tmp_path):
    camera = FakeCamera(frames=[])
    with pytest.raises(VisionError, match="recorded nothing"):
        iv.record_video(str(tmp_path / "q.avi"), stop=lambda: True, camera=camera)
    assert camera.released


# ---------- reading and analysing videos ----------

def make_video(path, seconds=3.0, fps=10, color=(255, 0, 0)):
    import cv2
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for _ in range(int(seconds * fps)):
        writer.write(np.full((48, 64, 3), color, np.uint8))
    writer.release()
    return path


def test_iter_frames_samples_at_the_analysis_rate_and_gives_rgb(tmp_path):
    path = make_video(str(tmp_path / "v.avi"))                     # 30 frames at 10 fps, BGR blue
    frames = list(iv.iter_frames(path, analysis_fps=5))
    assert len(frames) == 15 and frames[0][0] == 0.0 and frames[1][0] == pytest.approx(0.2)
    assert frames[0][1].shape == (48, 64, 3)
    assert frames[0][1][24, 32, 2] > 200 and frames[0][1][24, 32, 0] < 60     # blue in BGR is blue in RGB (B channel last)


def test_missing_or_unreadable_video(tmp_path):
    with pytest.raises(VisionError, match="not found"):
        list(iv.iter_frames(str(tmp_path / "nope.avi")))
    bad = tmp_path / "bad.avi"
    bad.write_bytes(b"not a video")
    with pytest.raises(VisionError, match="Could not open"):
        list(iv.iter_frames(str(bad)))


class FakeDetector:
    closed = False

    def close(self):
        self.closed = True


def test_analyze_video_with_fake_detectors(tmp_path, monkeypatch):
    path = make_video(str(tmp_path / "v.avi"))                     # 15 sampled frames over 3 s
    detectors, seen = (FakeDetector(), FakeDetector()), []

    def fake_detect(face_detector, pose_detector, rgb, ms):
        seen.append(ms)
        if len(seen) <= 10:                                         # first 10 frames: looking at the camera, upright
            return face(), {}, body()
        return None, {}, None                                       # then the person leaves the picture
    monkeypatch.setattr(iv, "_make_detectors", lambda: detectors)
    monkeypatch.setattr(iv, "_detect", fake_detect)
    s = iv.analyze_video(path)
    assert s["frames_analyzed"] == 15 and s["duration_seconds"] == 3.0
    assert s["eye_contact_percent"] == pytest.approx(66.7, abs=0.1) and s["face_detected_percent"] == pytest.approx(66.7, abs=0.1)
    assert s["posture_upright_percent"] == 100.0
    assert s["longest_look_away_seconds"] == 1.0
    assert seen == sorted(set(seen)) and seen[0] == 0              # timestamps strictly increase, as MediaPipe requires
    assert all(d.closed for d in detectors)


def test_analyze_video_closes_detectors_when_a_frame_fails(tmp_path, monkeypatch):
    path = make_video(str(tmp_path / "v.avi"))
    detectors = (FakeDetector(), FakeDetector())
    monkeypatch.setattr(iv, "_make_detectors", lambda: detectors)
    monkeypatch.setattr(iv, "_detect", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(VisionError, match="failed on a frame"):
        iv.analyze_video(path)
    assert all(d.closed for d in detectors)


def test_analyze_video_without_mediapipe_models(tmp_path):
    with pytest.raises(VisionError, match="MediaPipe"):
        iv.analyze_video(make_video(str(tmp_path / "v.avi")))        # conftest blocks the real loader


# ---------- the model files ----------

def test_existing_model_file_is_not_downloaded_again(tmp_path, monkeypatch):
    monkeypatch.setattr(iv, "MODEL_DIR", str(tmp_path))
    (tmp_path / "face_landmarker.task").write_bytes(b"model")
    monkeypatch.setattr("urllib.request.urlretrieve", lambda *a: pytest.fail("must not download"))
    assert iv.ensure_model("face_landmarker.task") == str(tmp_path / "face_landmarker.task")


def test_model_is_downloaded_once_then_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(iv, "MODEL_DIR", str(tmp_path / "models"))
    calls = []

    def fake_download(url, dest):
        calls.append(url)
        open(dest, "wb").write(b"model")
    monkeypatch.setattr("urllib.request.urlretrieve", fake_download)
    path = iv.ensure_model("pose_landmarker_lite.task")
    assert open(path, "rb").read() == b"model" and calls == [iv.MODEL_URLS["pose_landmarker_lite.task"]]
    iv.ensure_model("pose_landmarker_lite.task")
    assert len(calls) == 1


def test_failed_download_tells_how_to_fix_it(tmp_path, monkeypatch):
    monkeypatch.setattr(iv, "MODEL_DIR", str(tmp_path))

    def offline(url, dest):
        raise OSError("no internet")
    monkeypatch.setattr("urllib.request.urlretrieve", offline)
    with pytest.raises(VisionError, match="storage.googleapis.com.*face_landmarker.task"):
        iv.ensure_model("face_landmarker.task")
    assert not (tmp_path / "face_landmarker.task").exists()


# ---------- the glue to MediaPipe's real result types (no model needed) ----------

def test_detect_reads_mediapipe_result_objects():
    pytest.importorskip("mediapipe")
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.components.containers import category, landmark
    cat, point = category.Category, landmark.NormalizedLandmark

    class FakeFace:
        def __init__(self, found):
            self.found = found

        def detect_for_video(self, image, ms):
            assert image.width == 64 and image.height == 48 and ms == 7
            return vision.FaceLandmarkerResult(
                face_landmarks=[[point(x=.5, y=.5, z=0)] * 478] if self.found else [],
                face_blendshapes=[[cat(index=0, score=.9, display_name="", category_name="mouthSmileLeft")]] if self.found else [],
                facial_transformation_matrixes=[])

    class FakePose:
        def __init__(self, found):
            self.found = found

        def detect_for_video(self, image, ms):
            return vision.PoseLandmarkerResult(
                pose_landmarks=[[point(x=.5, y=.5, z=0, visibility=.9)] * 33] if self.found else [],
                pose_world_landmarks=[], segmentation_masks=None)
    frame = np.zeros((48, 64, 3), np.uint8)
    face_points, shapes, pose_points = iv._detect(FakeFace(True), FakePose(True), frame, 7)
    assert len(face_points) == 478 and shapes == {"mouthSmileLeft": pytest.approx(0.9)}
    assert len(pose_points) == 33 and pose_points[0].visibility == pytest.approx(0.9)
    assert iv._detect(FakeFace(False), FakePose(False), frame, 7) == (None, {}, None)
