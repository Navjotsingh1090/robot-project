import hashlib
import os
import re
import time

import cv2

from face_robot import config


def _sanitize_filename(name):
    safe = re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE).strip("_")
    return safe or "user"


def delete_user_face_images(raw_name: str) -> int:
    """
    Delete stored face crops for a user under FACE_FILES_DIR.
    Matches files like '<SanitizedName>_*.jpg' (case-insensitive).
    Returns number of deleted files.
    """
    base = _sanitize_filename(str(raw_name or "").strip())
    directory = config.FACE_FILES_DIR
    try:
        entries = os.listdir(directory)
    except FileNotFoundError:
        return 0

    pattern = re.compile(rf"^{re.escape(base)}_.*\.jpg$", re.IGNORECASE)
    deleted = 0
    for filename in entries:
        if not pattern.match(filename):
            continue
        path = os.path.join(directory, filename)
        try:
            os.remove(path)
            deleted += 1
        except FileNotFoundError:
            continue
    return deleted


def map_face_to_frame_rect(top, right, bottom, left, rgb_shape, frame_shape):
    """Map face box from processed RGB (possibly scaled) back to BGR frame pixel coords."""
    fh, fw = frame_shape[:2]
    rh, rw = rgb_shape[:2]
    if rw <= 0 or rh <= 0:
        return 0, fw, fh, 0

    sx = fw / rw
    sy = fh / rh
    left_i = max(0, int(left * sx))
    right_i = min(fw, int(round(right * sx)))
    top_i = max(0, int(top * sy))
    bottom_i = min(fh, int(round(bottom * sy)))
    return top_i, right_i, bottom_i, left_i


def save_enrollment_reference(bgr_frame, face_location, rgb_shape, person_name):
    """
    Crop the face on the full-resolution frame, write JPEG under FACE_FILES_DIR,
    return (path, sha256_hex) or (None, None) on failure.
    """
    top, right, bottom, left = face_location
    top, right, bottom, left = map_face_to_frame_rect(
        top, right, bottom, left, rgb_shape, bgr_frame.shape
    )
    if bottom <= top or right <= left:
        return None, None

    os.makedirs(config.FACE_FILES_DIR, exist_ok=True)
    stamp = int(time.time() * 1000)
    base = _sanitize_filename(person_name)
    path = os.path.join(config.FACE_FILES_DIR, f"{base}_{stamp}.jpg")
    crop = bgr_frame[top:bottom, left:right]
    if crop.size == 0:
        return None, None

    cv2.imwrite(path, crop)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return path, digest
