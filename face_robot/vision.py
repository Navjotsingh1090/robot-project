import cv2
import face_recognition
import numpy as np

from face_robot import config


def iou(a, b):
    """IoU for (top, right, bottom, left) boxes."""
    at, ar, ab, al = a
    bt, br, bb, bl = b

    inter_left = max(al, bl)
    inter_right = min(ar, br)
    inter_top = max(at, bt)
    inter_bottom = min(ab, bb)

    iw = max(0, inter_right - inter_left)
    ih = max(0, inter_bottom - inter_top)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0, ar - al) * max(0, ab - at)
    area_b = max(0, br - bl) * max(0, bb - bt)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def open_camera():
    candidates = [config.CAMERA_INDEX]
    if config.ALLOW_CAMERA_FALLBACK:
        candidates.extend([0, 1, 2, 3, 4, 5])
    tried = set()
    for idx in candidates:
        if idx in tried:
            continue
        tried.add(idx)
        print(f"Trying camera index {idx}...")
        cam = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cam.isOpened():
            ok, _ = cam.read()
            if ok:
                print(f"✅ Camera opened at index {idx}")
                return cam
        cam.release()
    if config.ALLOW_CAMERA_FALLBACK:
        print("❌ Could not open any configured or fallback camera.")
    else:
        print(
            f"❌ Could not open configured camera index {config.CAMERA_INDEX}. "
            "Set CAMERA_INDEX correctly or set ALLOW_CAMERA_FALLBACK=1."
        )
    return None


def configure_capture(video):
    if video is None:
        return
    video.set(3, config.CAMERA_WIDTH)
    video.set(4, config.CAMERA_HEIGHT)
    video.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def preprocess_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def filter_face_locations(face_locations, min_face_size):
    valid_faces = []
    for top, right, bottom, left in face_locations:
        if (right - left) >= min_face_size and (bottom - top) >= min_face_size:
            valid_faces.append((top, right, bottom, left))
    return valid_faces


def detect_faces(frame):
    scaled = cv2.resize(frame, (0, 0), fx=config.FRAME_SCALE, fy=config.FRAME_SCALE)
    rgb = preprocess_frame(scaled)
    face_locations = face_recognition.face_locations(rgb, model=config.FACE_DETECTION_MODEL)
    valid_faces = filter_face_locations(face_locations, config.MIN_FACE_SIZE)

    if valid_faces or not config.DISTANT_FACE_RETRY:
        return rgb, valid_faces

    retry_scaled = cv2.resize(frame, (0, 0), fx=config.FAR_FRAME_SCALE, fy=config.FAR_FRAME_SCALE)
    retry_rgb = preprocess_frame(retry_scaled)
    retry_faces = face_recognition.face_locations(retry_rgb, model=config.FACE_DETECTION_MODEL)
    retry_valid_faces = filter_face_locations(retry_faces, config.MIN_FAR_FACE_SIZE)
    return retry_rgb, retry_valid_faces


def is_sharp_enough(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= config.MIN_BLUR_SCORE


def pick_primary_face(encodings, face_locations):
    """Return (encoding, face_location) for the largest face, or (None, None)."""
    if not encodings or not face_locations:
        return None, None

    largest_index = 0
    largest_area = -1
    for index, (top, right, bottom, left) in enumerate(face_locations):
        area = max(0, right - left) * max(0, bottom - top)
        if area > largest_area:
            largest_area = area
            largest_index = index

    return encodings[largest_index], face_locations[largest_index]


def match_known_user(encoding, known_profiles, known_samples):
    """
    Compare one face encoding to known users. Uses the same L2 distance as
    face_recognition.face_distance, batched with NumPy for lower overhead.
    """
    if not known_profiles:
        return "Unknown"

    enc = np.asarray(encoding, dtype=np.float64).ravel()
    tol_ext = config.TOLERANCE + 0.03
    best_name = "Unknown"
    best_score = None

    for candidate_name, profile_encoding in known_profiles.items():
        prof = np.asarray(profile_encoding, dtype=np.float64).ravel()
        profile_distance = float(np.linalg.norm(prof - enc))
        samples_arr = np.asarray(known_samples[candidate_name], dtype=np.float64)
        if samples_arr.ndim == 1:
            samples_arr = samples_arr.reshape(1, -1)
        sample_distances = np.linalg.norm(samples_arr - enc, axis=1)
        support_matches = int(np.sum(sample_distances < tol_ext))
        score = profile_distance - (0.015 * min(support_matches, 3))

        if best_score is None or score < best_score:
            best_score = score
            best_name = candidate_name

    if best_score is not None and best_score < config.TOLERANCE:
        return best_name
    return "Unknown"
