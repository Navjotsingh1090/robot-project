import sys
import time

import face_recognition
import numpy as np

from face_robot import config
from face_robot import database
from face_robot import face_storage
from face_robot import speech_input
from face_robot import vision
from face_robot import voice


def run():
    voice.init_voice()
    database.init_db()
    voice.speak("System ready")

    speech_input.init_microphone()

    video = vision.open_camera()
    vision.configure_capture(video)
    time.sleep(2)

    if video is None or not video.isOpened():
        print("❌ Camera error")
        sys.exit(1)

    print("✅ System running")

    known_profiles, known_samples = database.load_users()

    frame_count = 0
    match_counts = {}
    unknown_count = 0
    last_enrollment_time = 0.0

    # Greeting is "once per continuous presence".
    greeted_active = set()
    last_present = {}
    greeted_groups = set()
    last_group_present = {}

    # Cache to avoid recomputing encodings for stable faces.
    # Each entry: {"box": (t,r,b,l), "name": str, "ts": float}
    track_cache = []
    _TRACK_CACHE_MAX = 48

    try:
        while True:
            try:
                ret, frame = video.read()
                if not ret:
                    # Avoid busy-spinning when the device stalls or returns no frame.
                    time.sleep(0.02)
                    continue

                frame_count += 1

                if frame_count % config.PROCESS_EVERY_N_FRAMES != 0:
                    continue

                rgb, valid_faces = vision.detect_faces(frame)
                if not valid_faces:
                    match_counts.clear()
                    unknown_count = 0
                    track_cache.clear()
                    continue

                now = time.time()

                # Drop stale cache entries.
                if config.TRACK_CACHE_SECONDS > 0:
                    cutoff = now - config.TRACK_CACHE_SECONDS
                    track_cache = [e for e in track_cache if e["ts"] >= cutoff]
                else:
                    track_cache = []
                if len(track_cache) > _TRACK_CACHE_MAX:
                    track_cache.sort(key=lambda e: e["ts"])
                    track_cache = track_cache[-_TRACK_CACHE_MAX:]

                # Determine which faces can be labeled from cache.
                names = ["Unknown"] * len(valid_faces)
                needs_encoding = []
                for idx, box in enumerate(valid_faces):
                    best = None
                    best_iou = 0.0
                    for entry in track_cache:
                        score = vision.iou(box, entry["box"])
                        if score > best_iou:
                            best_iou = score
                            best = entry
                    if best is not None and best_iou >= config.TRACK_IOU_THRESHOLD:
                        names[idx] = best["name"]
                        best["ts"] = now
                        best["box"] = box
                    else:
                        needs_encoding.append(idx)

                # Compute encodings only for faces we couldn't track.
                if needs_encoding:
                    boxes = [valid_faces[i] for i in needs_encoding]
                    encs = face_recognition.face_encodings(
                        rgb,
                        boxes,
                        num_jitters=config.FACE_ENCODING_JITTERS,
                    )
                    # face_recognition can return fewer encodings than boxes in edge cases.
                    for rel_i, enc in enumerate(encs):
                        face_idx = needs_encoding[rel_i]
                        name = vision.match_known_user(enc, known_profiles, known_samples)
                        names[face_idx] = name
                        track_cache.append({"box": valid_faces[face_idx], "name": name, "ts": now})

                # Reset greeting eligibility after someone leaves the frame.
                # IMPORTANT: do this BEFORE updating last_present with the current frame,
                # otherwise a re-entering person would never be considered "absent".
                for name in list(greeted_active):
                    if now - last_present.get(name, 0.0) > config.EXIT_RESET_SECONDS:
                        greeted_active.discard(name)
                        match_counts.pop(name, None)

                for gkey in list(greeted_groups):
                    if now - last_group_present.get(gkey, 0.0) > config.EXIT_RESET_SECONDS:
                        greeted_groups.discard(gkey)

                present_known = {name for name in names if name != "Unknown"}
                for name in present_known:
                    last_present[name] = now

                group_greeting = None
                group_key = None

                if len(valid_faces) >= 4:
                    group_greeting = "Hello everyone"
                    group_key = "group_everyone"
                elif len(valid_faces) >= 2:
                    group_greeting = "Hello guys"
                    group_key = "group_guys"

                if group_greeting is not None:
                    match_counts.clear()
                    unknown_count = 0
                    last_group_present[group_key] = now
                    if group_key not in greeted_groups:
                        voice.speak_async(group_greeting)
                        greeted_groups.add(group_key)
                    continue

                for name in names:

                    if name != "Unknown":
                        unknown_count = 0
                        match_counts[name] = match_counts.get(name, 0) + 1
                        for candidate_name in list(match_counts.keys()):
                            if candidate_name != name:
                                match_counts[candidate_name] = 0

                        if (
                            match_counts[name] >= config.RECOGNITION_STREAK
                            and name not in greeted_active
                        ):
                            voice.speak_async(f"Hello {name}")
                            greeted_active.add(name)

                    else:
                        match_counts.clear()
                        unknown_count += 1
                        if (
                            now - last_enrollment_time > config.ENROLLMENT_GRACE_PERIOD
                            and unknown_count >= config.UNKNOWN_STREAK
                            and (
                                "unknown" not in last_present
                                or now - last_present["unknown"] > config.UNKNOWN_COOLDOWN
                            )
                        ):
                            last_present["unknown"] = now
                            voice.speak_async("Hello")

                            person_name = speech_input.get_name()

                            if person_name is None:
                                unknown_count = 0
                                continue

                            samples = []
                            face_file_hash = None

                            for _ in range(config.ENROLLMENT_SAMPLES):
                                ret, frame = video.read()
                                if not ret:
                                    time.sleep(0.05)
                                    continue
                                if not vision.is_sharp_enough(frame):
                                    time.sleep(0.05)
                                    continue

                                rgb_i, faces = vision.detect_faces(frame)
                                encs = face_recognition.face_encodings(
                                    rgb_i,
                                    faces,
                                    num_jitters=config.FACE_ENCODING_JITTERS,
                                )

                                primary_encoding, primary_box = vision.pick_primary_face(encs, faces)
                                if primary_encoding is not None:
                                    samples.append(primary_encoding)

                                if face_file_hash is None and primary_box is not None:
                                    _, face_file_hash = face_storage.save_enrollment_reference(
                                        frame,
                                        primary_box,
                                        rgb_i.shape,
                                        person_name,
                                    )

                                time.sleep(0.3)

                            if len(samples) > 0:
                                database.save_user(person_name, samples, face_file_hash=face_file_hash)

                                voice.speak_async(f"Nice to meet you {person_name}")

                                existing_samples = known_samples.get(person_name, [])
                                updated_samples = existing_samples + samples
                                known_samples[person_name] = updated_samples
                                known_profiles[person_name] = np.mean(
                                    np.array(updated_samples), axis=0
                                )

                                last_present[person_name] = time.time()
                                last_present["unknown"] = time.time()
                                last_enrollment_time = time.time()
                                unknown_count = 0

            except KeyboardInterrupt:
                return

    finally:
        video.release()
        database.close_db()
