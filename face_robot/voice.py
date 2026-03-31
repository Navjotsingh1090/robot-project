import contextlib
import os
import shutil
import subprocess
import tempfile
import threading

from face_robot import config
from face_robot.audio_utils import suppress_stderr

voice_ready = False
tts_engine_active = ""
audio_player_wav = None
tts_lock = threading.Lock()


def get_wav_player():
    for candidate in ("paplay", "aplay"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def play_wav(path):
    if audio_player_wav is None:
        raise RuntimeError("No paplay/aplay found for WAV playback.")

    command = [audio_player_wav, path]
    if os.path.basename(audio_player_wav) == "aplay":
        command = [audio_player_wav, "-q", path]

    with suppress_stderr():
        subprocess.run(command, check=True)


def _synthesize_espeak_to_wav(text):
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    if espeak is None:
        raise RuntimeError("espeak-ng not found. Install: sudo apt install espeak-ng")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name

    subprocess.run(
        [
            espeak,
            "-w",
            out_path,
            "-v",
            config.ESPEAK_VOICE,
            "-s",
            str(config.ESPEAK_WPM),
            text,
        ],
        check=True,
        capture_output=True,
    )
    return out_path


def _piper_binary_path():
    if config.PIPER_BINARY:
        return config.PIPER_BINARY
    return shutil.which("piper")


def _piper_model_ready():
    model = os.path.abspath(os.path.expanduser(config.PIPER_MODEL))
    if not os.path.isfile(model):
        return None
    json_sidecar = model + ".json"
    if not os.path.isfile(json_sidecar):
        return None
    return model


def _synthesize_piper_to_wav(text):
    binary = _piper_binary_path()
    if not binary:
        raise RuntimeError("piper executable not found (install Piper or set PIPER_BINARY).")

    model = _piper_model_ready()
    if model is None:
        raise RuntimeError("Piper model or .onnx.json missing — see startup message for download URL.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name

    cmd = [
        binary,
        "--model",
        model,
        "--output_file",
        out_path,
    ]
    if abs(config.PIPER_LENGTH_SCALE - 1.0) > 0.01:
        cmd.extend(["--length-scale", str(config.PIPER_LENGTH_SCALE)])

    subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        check=True,
        capture_output=True,
    )
    return out_path


def _init_espeak(reason_note=None):
    global voice_ready, tts_engine_active

    if shutil.which("espeak-ng") is None and shutil.which("espeak") is None:
        return False
    if audio_player_wav is None:
        return False
    voice_ready = True
    tts_engine_active = "espeak"
    msg = f"✅ TTS: espeak-ng + {os.path.basename(audio_player_wav)}"
    if reason_note:
        msg = f"{msg} ({reason_note})"
    print(msg)
    return True


def init_voice():
    global voice_ready, tts_engine_active, audio_player_wav

    if not config.ENABLE_VOICE:
        return

    audio_player_wav = get_wav_player()

    engine = config.TTS_ENGINE

    if engine == "espeak":
        if not _init_espeak():
            print("⚠️ Voice disabled: install espeak-ng and paplay/aplay.")
        return

    if engine == "piper":
        model = _piper_model_ready()
        binary = _piper_binary_path()
        if model and binary and audio_player_wav:
            voice_ready = True
            tts_engine_active = "piper"
            print(f"✅ TTS: Piper (neural, offline) + {os.path.basename(audio_player_wav)}")
            print(f"   model: {model}")
            return

        print(
            "⚠️ Piper not ready (natural offline voice). "
            "Ensure `piper` is on PATH and the model + .onnx.json exist.\n"
            "   Default model path: /opt/piper/models/en_US-lessac-medium.onnx\n"
            f"   Set PIPER_MODEL if yours differs. Currently: {config.PIPER_MODEL}\n"
            "   Voices: https://huggingface.co/rhasspy/piper-voices"
        )
        if not binary:
            print("   Piper binary: https://github.com/rhasspy/piper/releases (Linux asset)")
        if _init_espeak("fallback from Piper"):
            return
        print("⚠️ Voice disabled: Piper unavailable and espeak-ng not available.")
        return

    print(f"⚠️ Unknown TTS_ENGINE={engine!r}; use piper or espeak.")


def speak(text):
    print("Robot:", text)
    if not voice_ready or not config.ENABLE_VOICE:
        return
    try:
        with tts_lock:
            if tts_engine_active == "espeak":
                path = _synthesize_espeak_to_wav(text)
                try:
                    play_wav(path)
                finally:
                    with contextlib.suppress(FileNotFoundError):
                        os.remove(path)
            elif tts_engine_active == "piper":
                path = _synthesize_piper_to_wav(text)
                try:
                    play_wav(path)
                finally:
                    with contextlib.suppress(FileNotFoundError):
                        os.remove(path)
            else:
                raise RuntimeError("Only offline TTS engines are allowed (piper/espeak).")
    except Exception as e:
        print(f"⚠️ Voice playback failed: {e}")


def speak_async(text):
    threading.Thread(target=speak, args=(text,), daemon=True).start()
