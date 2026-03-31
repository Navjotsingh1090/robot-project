"""Settings from `.env` (via python-dotenv) or the process environment."""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

SUPPRESS_AUDIO_BACKEND_NOISE = os.getenv("SUPPRESS_AUDIO_BACKEND_NOISE", "1") == "1"
ENABLE_VOICE = os.getenv("ENABLE_VOICE", "1") == "1"

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "2"))
ALLOW_CAMERA_FALLBACK = os.getenv("ALLOW_CAMERA_FALLBACK", "1") == "1"
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))

MIC_INDEX = int(os.getenv("MIC_INDEX", "1"))
MIC_NAME_HINT = os.getenv("MIC_NAME_HINT", "fingers")
MIC_SAMPLE_RATE = int(os.getenv("MIC_SAMPLE_RATE", "16000"))
MIC_CHUNK_SIZE = int(os.getenv("MIC_CHUNK_SIZE", "1024"))
MIC_AMBIENT_DURATION = float(os.getenv("MIC_AMBIENT_DURATION", "0.1"))
MIC_MIN_ENERGY = int(os.getenv("MIC_MIN_ENERGY", "120"))

TOLERANCE = float(os.getenv("FACE_TOLERANCE", "0.42"))
FRAME_SCALE = float(os.getenv("FRAME_SCALE", "0.5"))
FAR_FRAME_SCALE = float(os.getenv("FAR_FRAME_SCALE", "0.75"))
PROCESS_EVERY_N_FRAMES = int(os.getenv("PROCESS_EVERY_N_FRAMES", "3"))
FACE_ENCODING_JITTERS = int(os.getenv("FACE_ENCODING_JITTERS", "2"))
MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", "35"))
MIN_FAR_FACE_SIZE = int(os.getenv("MIN_FAR_FACE_SIZE", "24"))
FACE_DETECTION_MODEL = os.getenv("FACE_DETECTION_MODEL", "hog")
DISTANT_FACE_RETRY = os.getenv("DISTANT_FACE_RETRY", "0") == "1"
MIN_BLUR_SCORE = float(os.getenv("MIN_BLUR_SCORE", "70"))

TRACK_CACHE_SECONDS = float(os.getenv("TRACK_CACHE_SECONDS", "1.0"))
TRACK_IOU_THRESHOLD = float(os.getenv("TRACK_IOU_THRESHOLD", "0.45"))
EXIT_RESET_SECONDS = float(os.getenv("EXIT_RESET_SECONDS", "5.0"))

NAME_TIMEOUT = float(os.getenv("NAME_TIMEOUT", "10"))
NAME_PHRASE_TIME_LIMIT = float(os.getenv("NAME_PHRASE_TIME_LIMIT", "8"))

STT_ENGINE = os.getenv("STT_ENGINE", "google").strip().lower()
SPEECH_LANGUAGE = os.getenv("SPEECH_LANGUAGE", "en-IN").strip()
SPEECH_ALT_LANGUAGE = os.getenv("SPEECH_ALT_LANGUAGE", "").strip()
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "/opt/vosk/models").strip()
VOSK_MAX_SECONDS = float(os.getenv("VOSK_MAX_SECONDS", "5.0"))
VOSK_SILENCE_TIMEOUT = float(os.getenv("VOSK_SILENCE_TIMEOUT", "1.5"))
VOSK_GRAMMAR = os.getenv("VOSK_GRAMMAR", "").strip()

RECOGNITION_STREAK = int(os.getenv("RECOGNITION_STREAK", "2"))
UNKNOWN_STREAK = int(os.getenv("UNKNOWN_STREAK", "2"))
UNKNOWN_COOLDOWN = float(os.getenv("UNKNOWN_COOLDOWN", "15"))
ENROLLMENT_GRACE_PERIOD = float(os.getenv("ENROLLMENT_GRACE_PERIOD", "30"))
ENROLLMENT_SAMPLES = int(os.getenv("ENROLLMENT_SAMPLES", "5"))

TTS_ENGINE = os.getenv("TTS_ENGINE", "piper").strip().lower()
ESPEAK_VOICE = os.getenv("ESPEAK_VOICE", "en").strip()
ESPEAK_WPM = int(os.getenv("ESPEAK_WPM", "150"))
PIPER_BINARY = os.getenv("PIPER_BINARY", "").strip()
PIPER_MODEL = os.getenv(
    "PIPER_MODEL",
    "/opt/piper/models/en_US-lessac-medium.onnx",
).strip()
PIPER_LENGTH_SCALE = float(os.getenv("PIPER_LENGTH_SCALE", "0.9"))

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "robot")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "robot")

FACE_FILES_DIR = os.getenv("FACE_FILES_DIR", "data/faces")
