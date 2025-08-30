# voice_service.py
import os
from gtts import gTTS
from faster_whisper import WhisperModel

# Config via environment variables (defaults)
_MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")    # small, base, medium, large-v2
_WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # "cpu" or "cuda"
_WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

print(f"[voice_service] Loading faster-whisper model: {_MODEL_SIZE}.en on {_WHISPER_DEVICE} ...")
_whisper_model = WhisperModel(_MODEL_SIZE + ".en", device=_WHISPER_DEVICE, compute_type=_WHISPER_COMPUTE_TYPE)
print("[voice_service] faster-whisper loaded.")


def transcribe_audio(file_path, beam_size=5):
    """
    Transcribe an audio file using faster-whisper. Accepts wav/webm/ogg/mp3.
    Returns transcription (string) or '' on failure.
    """
    try:
        segments, _ = _whisper_model.transcribe(file_path, beam_size=beam_size)
        transcription = " ".join([seg.text for seg in segments]).strip()
        return transcription
    except Exception as e:
        print("[voice_service] Transcription error:", e)
        return ""


def play_text_to_file(text, out_file_path, language='en', slow=False):
    """
    Generate an MP3 using gTTS and save directly to out_file_path.
    """
    try:
        safe_text = text if (text and text.strip()) else "Sorry, I don't have a spoken response."
        os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
        tts = gTTS(text=safe_text, lang=language, slow=slow)
        tts.save(out_file_path)
        return out_file_path
    except Exception as e:
        print(f"[voice_service] Error generating TTS: {e}")
        return None
