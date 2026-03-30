import sys
import os
import pickle
import threading
import time
import contextlib

import mysql.connector

DELETE_MODE = len(sys.argv) >= 3 and sys.argv[1] == "remove-user"

if not DELETE_MODE:
    import cv2
    import numpy as np
    import speech_recognition as sr

    try:
        import pyttsx3
    except ImportError:
        print("❌ Missing dependency: pyttsx3")
        print("Install it in your active environment and rerun.")
        sys.exit(1)

    try:
        import face_recognition
    except ImportError:
        print("❌ Missing dependency: face_recognition")
        print("Install it in your active environment and rerun.")
        sys.exit(1)

# ---------------- CONFIG ----------------
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "4"))
ALLOW_CAMERA_FALLBACK = os.getenv("ALLOW_CAMERA_FALLBACK", "0") == "1"
MIC_INDEX = int(os.getenv("MIC_INDEX", "1"))
MIC_NAME_HINT = os.getenv("MIC_NAME_HINT", "fingers")
TOLERANCE = float(os.getenv("FACE_TOLERANCE", "0.42"))
COOLDOWN = 10
ENABLE_VOICE = (os.getenv("ENABLE_VOICE", "1") == "1") and not DELETE_MODE
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
FRAME_SCALE = float(os.getenv("FRAME_SCALE", "0.5"))
FAR_FRAME_SCALE = float(os.getenv("FAR_FRAME_SCALE", "0.75"))
PROCESS_EVERY_N_FRAMES = int(os.getenv("PROCESS_EVERY_N_FRAMES", "3"))
FACE_ENCODING_JITTERS = int(os.getenv("FACE_ENCODING_JITTERS", "2"))
MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", "35"))
MIN_FAR_FACE_SIZE = int(os.getenv("MIN_FAR_FACE_SIZE", "24"))
NAME_TIMEOUT = float(os.getenv("NAME_TIMEOUT", "10"))
NAME_PHRASE_TIME_LIMIT = float(os.getenv("NAME_PHRASE_TIME_LIMIT", "8"))
MIC_SAMPLE_RATE = int(os.getenv("MIC_SAMPLE_RATE", "16000"))
MIC_CHUNK_SIZE = int(os.getenv("MIC_CHUNK_SIZE", "1024"))
MIC_AMBIENT_DURATION = float(os.getenv("MIC_AMBIENT_DURATION", "1.0"))
MIC_MIN_ENERGY = int(os.getenv("MIC_MIN_ENERGY", "120"))
MIC_DYNAMIC_RATIO = float(os.getenv("MIC_DYNAMIC_RATIO", "1.25"))
MIC_PAUSE_THRESHOLD = float(os.getenv("MIC_PAUSE_THRESHOLD", "1.2"))
MIC_NON_SPEAKING_DURATION = float(os.getenv("MIC_NON_SPEAKING_DURATION", "0.6"))
SPEECH_LANGUAGE = os.getenv("SPEECH_LANGUAGE", "en-IN")
SPEECH_ALT_LANGUAGE = os.getenv("SPEECH_ALT_LANGUAGE", "en-US")
POST_TTS_PAUSE = float(os.getenv("POST_TTS_PAUSE", "0.7"))
RECOGNITION_STREAK = int(os.getenv("RECOGNITION_STREAK", "2"))
UNKNOWN_STREAK = int(os.getenv("UNKNOWN_STREAK", "2"))
UNKNOWN_COOLDOWN = float(os.getenv("UNKNOWN_COOLDOWN", "15"))
ENROLLMENT_GRACE_PERIOD = float(os.getenv("ENROLLMENT_GRACE_PERIOD", "30"))
ENROLLMENT_SAMPLES = int(os.getenv("ENROLLMENT_SAMPLES", "5"))
MIN_BLUR_SCORE = float(os.getenv("MIN_BLUR_SCORE", "70"))
FACE_DETECTION_MODEL = os.getenv("FACE_DETECTION_MODEL", "hog")
DISTANT_FACE_RETRY = os.getenv("DISTANT_FACE_RETRY", "0") == "1"

# ---------------- FIX AUDIO WARNINGS ----------------
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# ---------------- VOICE ----------------
engine = None
tts_lock = threading.Lock()


@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w", encoding="utf-8") as null_stream:
        with contextlib.redirect_stderr(null_stream):
            yield


if ENABLE_VOICE:
    try:
        with suppress_stderr():
            engine = pyttsx3.init('espeak')
        engine.setProperty('rate', 150)
    except Exception as e:
        print(f"⚠️ Voice engine unavailable: {e}")

def speak(text):
    print("Robot:", text)
    if engine is None or not ENABLE_VOICE:
        return
    try:
        # pyttsx3 is not thread-safe; serialize all TTS calls.
        with tts_lock:
            with suppress_stderr():
                engine.say(text)
                engine.runAndWait()
    except Exception as e:
        print(f"⚠️ Voice playback failed: {e}")

def speak_async(text):
    threading.Thread(target=speak, args=(text,), daemon=True).start()

# ---------------- DATABASE ----------------
db = None
cursor = None


def init_db():
    global db, cursor
    try:
        db = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", "Robo@1234"),
            database=os.getenv("DB_NAME", "robot"),
        )
        cursor = db.cursor()
        print("✅ Database connected")
    except Exception as e:
        db = None
        cursor = None
        print(f"⚠️ Database unavailable: {e}")
        print("Running without persistence.")


init_db()
speak("System ready")

def save_user(name, encodings):
    if cursor is None or db is None:
        print("⚠️ Skipping save: database not connected.")
        return
    for enc in encodings:
        data = pickle.dumps(enc)
        cursor.execute(
            "INSERT INTO users (name, encoding) VALUES (%s, %s)",
            (name, data)
        )
    db.commit()

def remove_user(name):
    if cursor is None or db is None:
        print("⚠️ Skipping delete: database not connected.")
        return 0

    clean_name = normalize_name(name)
    if clean_name is None:
        print("⚠️ Invalid user name.")
        return 0

    cursor.execute("DELETE FROM users WHERE LOWER(name) = LOWER(%s)", (clean_name,))
    deleted_rows = cursor.rowcount
    db.commit()
    return deleted_rows

def load_users():
    if cursor is None:
        return {}, {}
    cursor.execute("SELECT name, encoding FROM users")
    rows = cursor.fetchall()

    user_samples = {}

    for row in rows:
        clean_name = normalize_name(row[0])
        if clean_name is None:
            continue
        user_samples.setdefault(clean_name, []).append(pickle.loads(row[1]))

    user_profiles = {
        name: np.mean(np.array(encodings), axis=0)
        for name, encodings in user_samples.items()
        if encodings
    }

    return user_profiles, user_samples

# ---------------- SPEECH ----------------
def normalize_name(raw_name):
    if raw_name is None:
        return None

    words = str(raw_name).strip().split()
    if not words:
        return None

    filler_words = {
        "my", "name", "is", "i", "am", "i'm", "this", "its", "it's",
        "it", "me", "called", "call",
    }

    cleaned_words = []
    for word in words:
        letters_only = ''.join(c for c in word if c.isalpha())
        if not letters_only:
            continue
        if letters_only.lower() in filler_words:
            continue
        cleaned_words.append(letters_only)

    if not cleaned_words:
        return None

    name = cleaned_words[0]
    if len(name) < 2:
        return None
    return name.capitalize()


if DELETE_MODE:
    deleted_rows = remove_user(sys.argv[2])
    if deleted_rows > 0:
        print(f"✅ Removed {deleted_rows} saved face record(s) for {sys.argv[2]}")
    else:
        print(f"ℹ️ No saved user found for {sys.argv[2]}")

    if db is not None:
        try:
            if cursor is not None:
                cursor.close()
        finally:
            db.close()
    sys.exit(0)


def resolve_microphone_index():
    try:
        mic_names = sr.Microphone.list_microphone_names()
    except Exception as e:
        print(f"⚠️ Could not list microphones: {e}")
        return MIC_INDEX

    if MIC_NAME_HINT:
        hint = MIC_NAME_HINT.lower()
        for index, mic_name in enumerate(mic_names):
            if hint in mic_name.lower():
                print(f"✅ Using microphone {index}: {mic_name}")
                return index

    if 0 <= MIC_INDEX < len(mic_names):
        print(f"✅ Using configured microphone {MIC_INDEX}: {mic_names[MIC_INDEX]}")
        return MIC_INDEX

    print("⚠️ Requested microphone index not available. Falling back to default microphone.")
    return None


ACTIVE_MIC_INDEX = resolve_microphone_index()


def get_name():
    r = sr.Recognizer()
    r.energy_threshold = MIC_MIN_ENERGY
    r.dynamic_energy_threshold = True
    r.dynamic_energy_adjustment_damping = 0.12
    r.dynamic_energy_ratio = MIC_DYNAMIC_RATIO
    r.pause_threshold = MIC_PAUSE_THRESHOLD
    r.phrase_threshold = 0.2
    r.non_speaking_duration = MIC_NON_SPEAKING_DURATION
    r.operation_timeout = NAME_TIMEOUT + NAME_PHRASE_TIME_LIMIT

    for attempt in range(3):
        try:
            # Speak before opening the microphone so the robot does not
            # accidentally record its own voice.
            speak("Tell me your name")
            time.sleep(POST_TTS_PAUSE)

            with sr.Microphone(
                device_index=ACTIVE_MIC_INDEX,
                chunk_size=MIC_CHUNK_SIZE,
            ) as source:
                # Give the recognizer a bit more time to adapt to the room.
                r.adjust_for_ambient_noise(source, duration=MIC_AMBIENT_DURATION)
                r.energy_threshold = max(r.energy_threshold * 0.85, MIC_MIN_ENERGY)
                print(f"Listening with energy threshold: {r.energy_threshold:.1f}")

                audio = r.listen(
                    source,
                    timeout=NAME_TIMEOUT,
                    phrase_time_limit=NAME_PHRASE_TIME_LIMIT,
                )

            heard_text = None
            recognition_languages = [SPEECH_LANGUAGE]
            if SPEECH_ALT_LANGUAGE and SPEECH_ALT_LANGUAGE != SPEECH_LANGUAGE:
                recognition_languages.append(SPEECH_ALT_LANGUAGE)

            last_error = None
            for language in recognition_languages:
                try:
                    heard_text = r.recognize_google(audio, language=language)
                    print(f"Speech heard ({language}): {heard_text}")
                    break
                except sr.UnknownValueError as e:
                    last_error = e

            if heard_text is None:
                raise last_error or sr.UnknownValueError()

            name = normalize_name(heard_text)
            if name is None:
                raise sr.UnknownValueError()

            speak_async(f"I heard {name}")
            return name

        except sr.UnknownValueError:
            print(f"Speech error: could not understand name on attempt {attempt + 1}")
            if attempt < 2:
                speak("Please say only your first name clearly")
        except sr.WaitTimeoutError:
            print(f"Speech error: listening timed out on attempt {attempt + 1}")
            if attempt < 2:
                speak("I did not hear anything")
        except Exception as e:
            print(f"Speech error: {e!r}")
            if attempt < 2:
                speak("Let's try again")

    speak_async("Could not hear")
    return None

# ---------------- CAMERA ----------------
def open_camera():
    candidates = [CAMERA_INDEX]
    if ALLOW_CAMERA_FALLBACK:
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
    if ALLOW_CAMERA_FALLBACK:
        print("❌ Could not open any configured or fallback camera.")
    else:
        print(
            f"❌ Could not open configured camera index {CAMERA_INDEX}. "
            "Set CAMERA_INDEX correctly or set ALLOW_CAMERA_FALLBACK=1."
        )
    return None


video = open_camera()
if video is not None:
    video.set(3, CAMERA_WIDTH)
    video.set(4, CAMERA_HEIGHT)
    video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

time.sleep(2)

if video is None or not video.isOpened():
    print("❌ Camera error")
    sys.exit(1)

print("✅ System running")

# ---------------- LOAD DATA ----------------
known_profiles, known_samples = load_users()

last_seen = {}
frame_count = 0
match_counts = {}
unknown_count = 0
last_enrollment_time = 0.0


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
    scaled = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
    rgb = preprocess_frame(scaled)
    face_locations = face_recognition.face_locations(rgb, model=FACE_DETECTION_MODEL)
    valid_faces = filter_face_locations(face_locations, MIN_FACE_SIZE)

    if valid_faces or not DISTANT_FACE_RETRY:
        return rgb, valid_faces

    retry_scaled = cv2.resize(frame, (0, 0), fx=FAR_FRAME_SCALE, fy=FAR_FRAME_SCALE)
    retry_rgb = preprocess_frame(retry_scaled)
    retry_faces = face_recognition.face_locations(retry_rgb, model=FACE_DETECTION_MODEL)
    retry_valid_faces = filter_face_locations(retry_faces, MIN_FAR_FACE_SIZE)
    return retry_rgb, retry_valid_faces


def is_sharp_enough(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= MIN_BLUR_SCORE


def select_primary_encoding(face_locations, encodings):
    if not encodings:
        return None

    largest_index = 0
    largest_area = -1
    for index, (top, right, bottom, left) in enumerate(face_locations):
        area = max(0, right - left) * max(0, bottom - top)
        if area > largest_area:
            largest_area = area
            largest_index = index

    return encodings[largest_index]


def match_known_user(encoding):
    if not known_profiles:
        return "Unknown"

    best_name = "Unknown"
    best_score = None

    for candidate_name, profile_encoding in known_profiles.items():
        profile_distance = face_recognition.face_distance([profile_encoding], encoding)[0]
        sample_distances = face_recognition.face_distance(known_samples[candidate_name], encoding)
        support_matches = int(np.sum(sample_distances < (TOLERANCE + 0.03)))
        score = profile_distance - (0.015 * min(support_matches, 3))

        if best_score is None or score < best_score:
            best_score = score
            best_name = candidate_name

    if best_score is not None and best_score < TOLERANCE:
        return best_name
    return "Unknown"

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = video.read()
    if not ret:
        continue

    frame_count += 1

    # Process fewer skipped frames so distant faces are not missed as easily.
    if frame_count % PROCESS_EVERY_N_FRAMES != 0:
        if cv2.waitKey(1) == 27:
            break
        continue

    rgb, valid_faces = detect_faces(frame)

    encodings = face_recognition.face_encodings(
        rgb,
        valid_faces,
        num_jitters=FACE_ENCODING_JITTERS,
    )

    # Skip if no face
    if len(encodings) == 0:
        match_counts.clear()
        unknown_count = 0
        if cv2.waitKey(1) == 27:
            break
        continue

    now = time.time()
    group_greeting = None
    group_key = None

    if len(encodings) >= 4:
        group_greeting = "Hello everyone"
        group_key = "group_everyone"
    elif len(encodings) >= 2:
        group_greeting = "Hello guys"
        group_key = "group_guys"

    if group_greeting is not None:
        match_counts.clear()
        unknown_count = 0
        if group_key not in last_seen or now - last_seen[group_key] > COOLDOWN:
            speak_async(group_greeting)
            last_seen[group_key] = now
        if cv2.waitKey(1) == 27:
            break
        continue

    for encoding in encodings:
        name = match_known_user(encoding)

        # -------- KNOWN --------
        if name != "Unknown":
            unknown_count = 0
            match_counts[name] = match_counts.get(name, 0) + 1
            for candidate_name in list(match_counts.keys()):
                if candidate_name != name:
                    match_counts[candidate_name] = 0

            if (
                match_counts[name] >= RECOGNITION_STREAK
                and (name not in last_seen or now - last_seen[name] > COOLDOWN)
            ):
                speak_async(f"Hello {name}")
                last_seen[name] = now

        # -------- UNKNOWN --------
        else:
            match_counts.clear()
            unknown_count += 1
            if (
                now - last_enrollment_time > ENROLLMENT_GRACE_PERIOD
                and
                unknown_count >= UNKNOWN_STREAK
                and ("unknown" not in last_seen or now - last_seen["unknown"] > UNKNOWN_COOLDOWN)
            ):
                last_seen["unknown"] = now
                speak_async("Hello")

                person_name = get_name()

                if person_name is None:
                    unknown_count = 0
                    continue

                samples = []
                speak_async("Look at camera")

                for _ in range(ENROLLMENT_SAMPLES):
                    ret, frame = video.read()
                    if not ret:
                        continue
                    if not is_sharp_enough(frame):
                        continue

                    rgb, faces = detect_faces(frame)
                    encs = face_recognition.face_encodings(
                        rgb,
                        faces,
                        num_jitters=FACE_ENCODING_JITTERS,
                    )

                    primary_encoding = select_primary_encoding(faces, encs)
                    if primary_encoding is not None:
                        samples.append(primary_encoding)

                    time.sleep(0.3)

                if len(samples) > 0:
                    save_user(person_name, samples)

                    speak_async(f"Nice to meet you {person_name}")

                    existing_samples = known_samples.get(person_name, [])
                    updated_samples = existing_samples + samples
                    known_samples[person_name] = updated_samples
                    known_profiles[person_name] = np.mean(np.array(updated_samples), axis=0)

                    last_seen[person_name] = time.time()
                    last_seen["unknown"] = time.time()
                    last_enrollment_time = time.time()
                    unknown_count = 0

    # OPTIONAL DISPLAY
    # cv2.imshow("AI Robot Vision", frame)

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()
if db is not None:
    try:
        if cursor is not None:
            cursor.close()
    finally:
        db.close()
