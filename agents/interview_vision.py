"""Agent 4, step 3: computer vision. Records the webcam during an answer (or
takes a pre-recorded video) and uses MediaPipe to estimate how much of the
time the candidate looked at the camera, how upright they sat, and whether they smiled.

    record_video(path, stop)       webcam -> video file (real-time pace, so video time = clock time)
    analyze_video(path)            video -> per-frame observations -> metrics
    frame_observation(...)         PURE: MediaPipe landmarks of ONE frame -> yes/no judgments
    summarize_frames(obs, secs)    PURE: observations -> percentages

MediaPipe, OpenCV and the model files are loaded only when used, so importing
this file is cheap and the tests never touch a camera or a model (tests/conftest.py).
The two MediaPipe model files (about 4 MB + 6 MB) download once to models/mediapipe/
(git-ignored); everything then runs on this computer and no video leaves it.

THE JUDGMENTS ARE HEURISTICS on MediaPipe's landmarks, not a trained "eye
contact" classifier. The thresholds below are starting values meant to be
checked against real recordings of your own setup (see README, Agent 4).
"""
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "mediapipe")
MODEL_URLS = {
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}

RECORD_FPS = 15                    # webcam recording rate
MAX_RECORD_SECONDS = 120
ANALYSIS_FPS = 5                   # frames per second given to MediaPipe (the rest are skipped)
MIN_FRAMES_FOR_METRICS = 3         # fewer analysed frames than this: the percentages are None

# ---- eye contact: head pointing at the camera AND eyes not turned away ----
# Face landmark indices (MediaPipe face mesh): 1 nose tip, 234 / 454 the two face edges,
# 10 top of the forehead, 152 chin.
NOSE, FACE_EDGE_A, FACE_EDGE_B, FOREHEAD, CHIN = 1, 234, 454, 10, 152
MAX_YAW_OFFSET = 0.15              # nose may be this far from the middle of the face width (0.5 = centred)
PITCH_RANGE = (0.40, 0.75)         # nose position between forehead (0) and chin (1); wide: webcams sit above/below
MAX_EYE_TURN = 0.45                # blendshape score (0-1) of any "eye looks left/right/up/down" above this = looking away
EYE_BLENDSHAPES = [f"eyeLook{d}{side}" for d in ("In", "Out", "Up", "Down") for side in ("Left", "Right")]
SMILE_BLENDSHAPES = ("mouthSmileLeft", "mouthSmileRight")
SMILE_SCORE = 0.4                  # average smile score above this = smiling

# ---- posture, from the pose landmarks (11/12 shoulders, 7/8 ears, 0 nose) ----
LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_EAR, RIGHT_EAR = 11, 12, 7, 8
MIN_VISIBILITY = 0.5
MAX_SHOULDER_TILT_DEGREES = 12.0   # shoulders more tilted than this = leaning
MIN_NECK_RATIO = 0.25              # (shoulder line - ear line) / shoulder width; smaller = head sunk / shoulders hunched
#                                    (a relaxed upright adult is roughly 0.35-0.45 from the front; a hunch is clearly lower)
MAX_HEAD_OFFSET = 0.35             # nose sideways from the shoulders' middle, in shoulder widths = leaning to one side


class VisionError(Exception):
    """Recording or analysis failed. The message says what to do."""


# ---------- the two pure steps ----------

def _xy(landmarks, index):
    p = landmarks[index]
    return float(p.x), float(p.y)


def judge_face(face_landmarks, blendshapes):
    """One frame's face -> {"eye_contact": bool, "smile": bool}.
    face_landmarks: the 478 landmarks (objects with .x .y). blendshapes: {name: score}."""
    nose_x, nose_y = _xy(face_landmarks, NOSE)
    edge_a, edge_b = _xy(face_landmarks, FACE_EDGE_A)[0], _xy(face_landmarks, FACE_EDGE_B)[0]
    top, chin = _xy(face_landmarks, FOREHEAD)[1], _xy(face_landmarks, CHIN)[1]
    width, height = abs(edge_b - edge_a), chin - top
    if width <= 0 or height <= 0:
        return {"eye_contact": False, "smile": False}
    yaw = (nose_x - min(edge_a, edge_b)) / width
    pitch = (nose_y - top) / height
    head_ok = abs(yaw - 0.5) <= MAX_YAW_OFFSET and PITCH_RANGE[0] <= pitch <= PITCH_RANGE[1]
    eyes_ok = max((blendshapes.get(name, 0.0) for name in EYE_BLENDSHAPES), default=0.0) <= MAX_EYE_TURN
    smile = sum(blendshapes.get(name, 0.0) for name in SMILE_BLENDSHAPES) / len(SMILE_BLENDSHAPES) >= SMILE_SCORE
    return {"eye_contact": bool(head_ok and eyes_ok), "smile": bool(smile)}


def judge_posture(pose_landmarks):
    """One frame's body -> {"upright": bool, "shoulder_tilt_degrees": float} or
    None when the shoulders or ears can't be seen well enough."""
    needed = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_EAR, RIGHT_EAR)
    if any((getattr(pose_landmarks[i], "visibility", None) or 0.0) < MIN_VISIBILITY for i in needed):
        return None
    (lx, ly), (rx, ry) = _xy(pose_landmarks, LEFT_SHOULDER), _xy(pose_landmarks, RIGHT_SHOULDER)
    ear_y = (_xy(pose_landmarks, LEFT_EAR)[1] + _xy(pose_landmarks, RIGHT_EAR)[1]) / 2
    width = abs(lx - rx)
    if width <= 0:
        return None
    tilt = abs(float(np.degrees(np.arctan2(ly - ry, lx - rx))))
    tilt = min(tilt, 180.0 - tilt)                      # a shoulder line is the same line from either end
    neck = ((ly + ry) / 2 - ear_y) / width
    offset = abs(_xy(pose_landmarks, 0)[0] - (lx + rx) / 2) / width
    upright = tilt <= MAX_SHOULDER_TILT_DEGREES and neck >= MIN_NECK_RATIO and offset <= MAX_HEAD_OFFSET
    return {"upright": bool(upright), "shoulder_tilt_degrees": round(tilt, 1)}


def frame_observation(t, face_landmarks=None, blendshapes=None, pose_landmarks=None):
    """Everything judged in one frame. None for a part that wasn't found."""
    face = judge_face(face_landmarks, blendshapes or {}) if face_landmarks else None
    pose = judge_posture(pose_landmarks) if pose_landmarks else None
    return {"t": round(float(t), 2),
            "face_found": face is not None,
            "eye_contact": face["eye_contact"] if face else False,       # no face = not looking at the camera
            "smile": face["smile"] if face else None,
            "pose_found": pose is not None,
            "upright": pose["upright"] if pose else None,
            "shoulder_tilt_degrees": pose["shoulder_tilt_degrees"] if pose else None}


def percent(count, total):
    return round(100.0 * count / total, 1) if total else None


def summarize_frames(observations, duration_seconds):
    """Per-frame observations -> the answer's vision metrics.
    eye_contact_percent counts every analysed frame (a frame with no face counts
    as not looking). posture_upright_percent counts only frames where the
    shoulders and ears were visible. Percentages are None when too few frames
    were analysed to say anything."""
    n = len(observations)
    enough = n >= MIN_FRAMES_FOR_METRICS
    faces = [o for o in observations if o["face_found"]]
    poses = [o for o in observations if o["pose_found"]]
    step = (duration_seconds / n) if n and duration_seconds else 0.0
    longest = run = 0
    for o in observations:
        run = 0 if o["eye_contact"] else run + 1
        longest = max(longest, run)
    tilts = [o["shoulder_tilt_degrees"] for o in poses]
    return {
        "duration_seconds": round(float(duration_seconds or 0), 2),
        "frames_analyzed": n,
        "face_detected_percent": percent(len(faces), n) if enough else None,
        "eye_contact_percent": percent(sum(o["eye_contact"] for o in observations), n) if enough else None,
        "longest_look_away_seconds": round(longest * step, 1) if enough else None,
        "pose_detected_percent": percent(len(poses), n) if enough else None,
        "posture_upright_percent": percent(sum(o["upright"] for o in poses), len(poses)) if enough and poses else None,
        "average_shoulder_tilt_degrees": round(sum(tilts) / len(tilts), 1) if enough and tilts else None,
        "smile_percent": percent(sum(bool(o["smile"]) for o in faces), len(faces)) if enough and faces else None,
    }


# ---------- recording ----------

def _open_camera(index=0):
    try:
        import cv2
    except ImportError as e:
        raise VisionError(f"Recording needs OpenCV (pip install -r requirements.txt): {e}")
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        camera.release()
        raise VisionError("Could not open the webcam. Check that it is connected and that no other program is using it.")
    return camera


def record_video(path, stop=None, max_seconds=MAX_RECORD_SECONDS, camera=None):
    """Records the webcam to a video file at RECORD_FPS until `stop()` returns
    True or max_seconds pass. Frames are written on a fixed clock (a slow
    camera repeats its last frame), so a frame's position in the file is its
    real time. Returns the duration in seconds."""
    import cv2
    camera = camera or _open_camera()
    writer, frames, last = None, 0, None
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    started = time.time()
    try:
        while True:
            elapsed = time.time() - started
            if elapsed >= max_seconds or (stop and stop()):
                break
            ok, frame = camera.read()
            if ok and frame is not None:
                last = frame
            if last is None:
                if elapsed > 5:
                    raise VisionError("The webcam gave no picture for 5 seconds.")
                time.sleep(0.02)
                continue
            if writer is None:
                height, width = last.shape[:2]
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), RECORD_FPS, (width, height))
                if not writer.isOpened():
                    raise VisionError(f"Could not write the video file {path}.")
            # write one frame per tick of the clock that has passed
            due = int((time.time() - started) * RECORD_FPS) + 1
            while frames < due:
                writer.write(last)
                frames += 1
            time.sleep(max(0.0, started + frames / RECORD_FPS - time.time()))   # wait for the next tick
        if writer is not None and last is not None:
            due = int((time.time() - started) * RECORD_FPS)
            while frames < due:
                writer.write(last)
                frames += 1
    finally:
        camera.release()
        if writer is not None:
            writer.release()
    if frames == 0:
        raise VisionError("The webcam recorded nothing.")
    return frames / RECORD_FPS


# ---------- analysing a video ----------

def iter_frames(path, analysis_fps=ANALYSIS_FPS):
    """Yields (time in seconds, RGB frame) about `analysis_fps` times a second."""
    try:
        import cv2
    except ImportError as e:
        raise VisionError(f"Analysing video needs OpenCV (pip install -r requirements.txt): {e}")
    if not os.path.isfile(path):
        raise VisionError(f"Video file not found: {path}")
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        video.release()
        raise VisionError(f"Could not open the video {path}.")
    fps = video.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 else RECORD_FPS
    every = max(1, int(round(fps / analysis_fps)))
    index = 0
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            if index % every == 0:
                yield index / fps, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        video.release()


def video_duration(path):
    """Length of a video file in seconds (0.0 if it can't be read)."""
    import cv2
    video = cv2.VideoCapture(path)
    try:
        fps = video.get(cv2.CAP_PROP_FPS)
        count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        return round(count / fps, 2) if fps and fps > 0 and count and count > 0 else 0.0
    finally:
        video.release()


def ensure_model(name):
    """Path of a MediaPipe model file, downloading it on first use."""
    path = os.path.join(MODEL_DIR, name)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    import urllib.request
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        urllib.request.urlretrieve(MODEL_URLS[name], path + ".part")
        os.replace(path + ".part", path)
    except Exception as e:
        raise VisionError(f"Could not download the MediaPipe model {name} (needs internet once). Download "
                          f"{MODEL_URLS[name]} and save it as {path}. Error: {e}")
    return path


def _make_detectors():
    """(face_landmarker, pose_landmarker) in video mode. The only place that builds MediaPipe models."""
    try:
        import mediapipe  # noqa: F401
        from mediapipe.tasks.python import BaseOptions, vision
    except ImportError as e:
        raise VisionError(f"Video analysis needs MediaPipe (pip install -r requirements.txt): {e}")
    face_path, pose_path = ensure_model("face_landmarker.task"), ensure_model("pose_landmarker_lite.task")
    try:
        face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=face_path), running_mode=vision.RunningMode.VIDEO,
            num_faces=1, output_face_blendshapes=True))
        pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_path), running_mode=vision.RunningMode.VIDEO,
            num_poses=1))
    except Exception as e:
        raise VisionError(f"Could not start MediaPipe: {e}")
    return face, pose


def _detect(face_detector, pose_detector, rgb_frame, timestamp_ms):
    """MediaPipe on one RGB frame -> (face_landmarks, {blendshape: score}, pose_landmarks), each None if not found."""
    import mediapipe as mp
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
    face = face_detector.detect_for_video(image, timestamp_ms)
    pose = pose_detector.detect_for_video(image, timestamp_ms)
    landmarks = face.face_landmarks[0] if face.face_landmarks else None
    shapes = ({c.category_name: c.score for c in face.face_blendshapes[0]}
              if face.face_blendshapes else {})
    return landmarks, shapes, (pose.pose_landmarks[0] if pose.pose_landmarks else None)


def analyze_video(path):
    """Video file -> vision metrics (see summarize_frames). Works on a
    recording or a pre-recorded test video. A video with no person gives
    face_detected_percent 0, not an error."""
    detectors = _make_detectors()
    face_detector, pose_detector = detectors
    observations, last_ms = [], -1
    try:
        for t, frame in iter_frames(path):
            ms = max(int(t * 1000), last_ms + 1)          # MediaPipe needs strictly increasing times
            last_ms = ms
            try:
                face, shapes, pose = _detect(face_detector, pose_detector, frame, ms)
            except Exception as e:
                raise VisionError(f"MediaPipe failed on a frame: {e}")
            observations.append(frame_observation(t, face, shapes, pose))
    finally:
        for detector in detectors:
            try:
                detector.close()
            except Exception:
                pass
    return summarize_frames(observations, video_duration(path))
