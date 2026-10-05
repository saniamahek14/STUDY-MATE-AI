# core/voice.py
# Voice features, all offline:
#   transcribe()  speech -> text  (faster-whisper)
#   speak()       text   -> audio (pyttsx3, uses your computer's built-in voice)

import io
import os
import re
import tempfile
import wave

import numpy as np

_model = None


def _get_model():
    """Load Whisper only when first needed (keeps app start fast)."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        # "base" = small and fast. Use "small" for better accuracy.
        _model = WhisperModel("base", device="cpu", compute_type="int8")
    return _model


def _wav_bytes_to_array(data):
    """Decode WAV bytes (from the browser mic) into 16 kHz mono float32."""
    with wave.open(io.BytesIO(data), "rb") as w:
        channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        frames = w.readframes(w.getnframes())
    if width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {width}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != 16000 and len(audio) > 0:
        n = int(len(audio) * 16000 / rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, n),
                          np.arange(len(audio)), audio).astype(np.float32)
    return audio


def transcribe_wav_bytes(data):
    """Speech -> text for a mic recording (WAV bytes). No file needed."""
    audio = _wav_bytes_to_array(data)
    if len(audio) == 0:
        return ""
    segments, _info = _get_model().transcribe(audio)
    return " ".join(segment.text.strip() for segment in segments).strip()


def transcribe(path):
    """Convert an audio file (wav/mp3/m4a) into text."""
    segments, _info = _get_model().transcribe(path)
    return " ".join(segment.text.strip() for segment in segments).strip()


def _clean_for_speech(text, limit=800):
    """Remove markdown symbols so the voice doesn't read '**' and '#'."""
    text = re.sub(r"[*#`|_>]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def speak(text):
    """Return WAV audio bytes of the spoken text, or None if it fails."""
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        engine.save_to_file(_clean_for_speech(text), path)
        engine.runAndWait()
        engine.stop()
        with open(path, "rb") as f:
            data = f.read()
        os.remove(path)
        return data or None
    except Exception:
        return None
