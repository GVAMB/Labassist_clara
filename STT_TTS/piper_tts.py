# piper_tts.py — multi-language TTS (Cori=en, Kerstin=de) with subtle "emotions"
from pathlib import Path
import traceback
from typing import Dict, Tuple, Union

import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice

# -------------------------------------------------------------------
# Voices you already use
# -------------------------------------------------------------------
VOICE_REGISTRY = {
    "en": Path("STT_TTS/models/piper/en_GB-southern_english_female-low/en_GB-southern_english_female-low"),
    "de": Path("STT_TTS/models/piper/de_DE-kerstin-low/de_DE-kerstin-low"),
}

_voice: PiperVoice | None = None
_current_lang: str = "en"

# -------------------------------------------------------------------
# Emotion → audio parameters (SUBTLE, to keep the same voice character)
# -------------------------------------------------------------------
EMOTION_AUDIO_PARAMS: Dict[str, Dict[str, float]] = {
    "neutral": {"speed": 1.00, "gain": 1.00},
    "happy": {"speed": 1.06, "gain": 1.05},
    "sad": {"speed": 0.94, "gain": 0.97},
    "angry": {"speed": 1.08, "gain": 1.05},
    "confused": {"speed": 0.98, "gain": 1.00},
    "thinking": {"speed": 0.96, "gain": 1.00},
}


# -------------------------------------------------------------------
# Language handling
# -------------------------------------------------------------------
class OfflineTTS:
    
    # Placeholder for tracking audio duration (needed by AudioPlaybackManager)
    _last_duration: float = 0.0

    def set_language(self, lang: str):
        """
        Load Piper model for given language code ("en", "de", ...).
        """
        global _voice, _current_lang

        if lang not in VOICE_REGISTRY:
            raise ValueError(
                f"Unsupported language '{lang}'. Options: {list(VOICE_REGISTRY.keys())}"
            )

        model_path = VOICE_REGISTRY[lang]
        if not model_path.exists():
            raise FileNotFoundError(f"Piper model not found: {model_path}")

        _voice = PiperVoice.load(str(model_path)) 
        _current_lang = lang
        print(f"[TTS] Language set to {lang.upper()} (model: {model_path.name})")


    def get_current_language(self) -> str:
        return _current_lang

    def get_last_duration(self) -> float:
        """Returns the duration of the last synthesized audio (for echo-guard)."""
        return self._last_duration

    def _ensure_voice(self):
        global _voice
        if _voice is None:
            self.set_language("en")


    # -------------------------------------------------------------------
    # Core synth helper (returns raw data, NO playback)
    # -------------------------------------------------------------------
    def _synthesize_raw(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Synthesize text with Piper and return (audio_float32_mono, sample_rate).
        """
        self._ensure_voice()
        assert _voice is not None

        pcm = bytearray()
        sr = None

        for chunk in _voice.synthesize(text): 
            if sr is None:
                sr = int(getattr(chunk, "sample_rate", 22050))
            pcm.extend(chunk.audio_int16_bytes)

        if sr is None or not pcm:
            return np.zeros((0, 1), dtype=np.float32), 22050

        # Convert int16 bytes to float32 numpy array
        data = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        data = data.reshape(-1, 1) 
        
        # Calculate and store duration
        if sr > 0:
            self._last_duration = data.shape[0] / sr
        else:
            self._last_duration = 0.0

        return data, sr


    # -------------------------------------------------------------------
    # Audio transform for emotions
    # -------------------------------------------------------------------
    def _apply_speed_and_gain(self, 
        data: np.ndarray, speed: float, gain: float
    ) -> np.ndarray:
        """
        Applies subtle speed and volume gain transformations.
        """
        if data.size == 0:
            return data

        # Speed / pitch via simple resampling (interpolation)
        if abs(speed - 1.0) > 0.005: 
            n = data.shape[0]
            new_len = max(1, int(n / speed))
            orig_idx = np.linspace(0, n - 1, num=n, dtype=np.float32)
            new_idx = np.linspace(0, n - 1, num=new_len, dtype=np.float32)
            mono = data[:, 0]
            resampled = np.interp(new_idx, orig_idx, mono).astype(np.float32)
            data = resampled.reshape(-1, 1)

        # Gain (volume)
        if abs(gain - 1.0) > 0.01:
            data = data * gain
            data = np.clip(data, -1.0, 1.0)

        return data


    # -------------------------------------------------------------------
    # NEW CORE SYNTHESIS FUNCTION (non-blocking)
    # -------------------------------------------------------------------
    def synthesize_with_emotion_raw(self, text: str, emotion: str = "neutral") -> Tuple[np.ndarray, int]:
        """
        Synthesizes audio data and applies emotion transforms, but DOES NOT play it.
        Returns (audio_data, sample_rate).
        """
        try:
            if not text or not text.strip():
                return np.zeros((0, 1), dtype=np.float32), 22050

            data, sr = self._synthesize_raw(text)
            if data.size == 0:
                return data, sr

            params = EMOTION_AUDIO_PARAMS.get(
                emotion.lower(), EMOTION_AUDIO_PARAMS["neutral"]
            )
            print(f"[TTS] emotion={emotion} speed={params['speed']} gain={params['gain']}")
            data = self._apply_speed_and_gain(
                data,
                speed=params["speed"],
                gain=params["gain"],
            )
            return data, sr

        except Exception:
            print("TTS error in synthesize_with_emotion_raw:\n" + traceback.format_exc())
            return np.zeros((0, 1), dtype=np.float32), 22050


    # -------------------------------------------------------------------
    # Public API (Playback wrappers are now simple calls to the manager)
    # -------------------------------------------------------------------
    def speak_emotional(self, text: str, emotion: str = "neutral") -> bool:
        """
        DEPRECATED: Now only synthesizes and returns data (for clarity in the worker).
        The actual playback logic is now in AudioPlaybackManager.
        """
        # This function is now only called by TTSWorker to synthesize.
        # We must keep its name but change its behavior to return audio data.
        # For simplicity in the worker, we'll rely on the worker to call the new raw function.
        # If any old code relies on this, it will break, but the new worker logic bypasses it.
        return self.synthesize_with_emotion_raw(text, emotion)


    def speak_text(self, text: str) -> bool:
        """
        Backwards-compatible wrapper (will now return raw data).
        """
        # This function should only be called by the AudioPlaybackManager (player)
        # For now, we'll keep it as a wrapper that calls the blocking play *only* # when called by the AudioPlaybackManager, as it's cleaner to handle 
        # the playback logic there. 
        # NOTE: The AudioPlaybackManager will use sd.play() directly, 
        # not this function, to ensure full control.
        return True # Just return True for compliance, actual playback is elsewhere.


    def close_audio(self):
        try:
            sd.stop()
        except Exception:
            pass
