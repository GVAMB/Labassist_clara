"""
stt_worker.py

Contains the STTWorker class.
Uses ClaraWakeApp + Vosk for offline STT, gated by:
- session_active_event
- ready_to_listen_event
- is_speaking_event
"""

import time
import threading
from queue import Queue
from typing import Optional
import Levenshtein
import speech_recognition as sr

from config import RunningFlag, READY_SIGNAL
from STT_TTS.ClaraWakeApp import ClaraWakeApp, SR_MIC_DEVICE_INDEX
from tts_worker import AudioPlaybackManager


class STTWorker(threading.Thread):
    """
    STT waits for:
      - session_active_event (conversation window)
      - 'ready' cue (ready_to_listen_event) from TTS pipeline,
    and then performs exactly ONE listen per cue.

    Language switching is delegated to ClaraWakeApp.maybe_switch_language.
    """

    def __init__(
        self,
        wake_app: ClaraWakeApp,
        input_queue: Queue,
        running_flag: RunningFlag,
        is_speaking_event: threading.Event,
        ready_to_listen_event: threading.Event,
        session_active_event: threading.Event,
        stt_listening_event: threading.Event,  # tracks when STT is actively listening
        llm_output_queue: Queue,
        audio_manager: Optional['AudioPlaybackManager'] = None,
    ):
        super().__init__(daemon=True)
        self.wake_app = wake_app
        self.input_queue = input_queue
        self.running_flag = running_flag
        self.is_speaking_event = is_speaking_event
        self.ready_to_listen_event = ready_to_listen_event
        self.session_active_event = session_active_event
        self.stt_listening_event = stt_listening_event
        self.llm_output_queue = llm_output_queue

        self.audio_manager = audio_manager
        # cache for current Vosk recognizer
        self.vosk_recognizer = None
        self.vosk_lang = None

    # ---------- recognizer + helpers ----------

    def _ensure_recognizer(self):
        """Make sure we have a Vosk recognizer for the current Clara language."""
        lang = getattr(self.wake_app, "current_lang", "en")
        if self.vosk_recognizer is None or lang != self.vosk_lang:
            _, self.vosk_recognizer = self.wake_app.load_stt(lang)
            self.vosk_lang = lang
            print(f"[STTWorker] Loaded Vosk recognizer for lang={lang}")

    def _handle_language_switch(self, text: str) -> bool:
        """
        Use ClaraWakeApp.maybe_switch_language to flip EN<->DE.

        Returns True if:
          - a language switch was detected
          - ClaraWakeApp.current_lang was updated
          - and this utterance should NOT go to the LLM.
        """
        if hasattr(self.wake_app, "_norm"):
            t_norm = self.wake_app._norm(text)
        else:
            t_norm = " ".join((text or "").lower().strip().split())

        cur_lang = getattr(self.wake_app, "current_lang", "en")
        new_lang = self.wake_app.maybe_switch_language(t_norm, cur_lang)

        if new_lang and new_lang != cur_lang:
            print(f"[STTWorker] Language switch requested: {cur_lang} -> {new_lang}")
            self.wake_app.current_lang = new_lang
            self.vosk_recognizer = None

            try:
                self._ensure_recognizer()
                print(f"[STTWorker] Vosk model successfully updated to {new_lang.upper()}.")
            except Exception as e:
                print(f"[STTWorker] ERROR loading new Vosk model: {e}")
                # Restore previous language if load failed
                self.wake_app.current_lang = cur_lang
                self.vosk_recognizer = None

            # Immediately allow the user to speak again after the beep
            self.ready_to_listen_event.set()
            return True

        return False

    def _is_self_echo(self, raw_user_text: str) -> bool:
        """
        (Kept for logging / debugging only.)
        Checks if the transcribed raw text is just an acoustic echo 
        of the robot's last spoken phrase.
        """
        if not self.audio_manager:
            return False

        last_tts_text = self.audio_manager.last_spoken_text

        def normalize(text):
            text = (text or "").lower().strip()
            text = (
                text.replace(".", "")
                .replace("?", "")
                .replace("!", "")
                .replace(",", "")
            )
            return text

        normalized_tts = normalize(last_tts_text)
        normalized_stt = normalize(raw_user_text)

        if not normalized_tts:
            return False

        if normalized_tts in normalized_stt or normalized_stt in normalized_tts:
            print(f"[STT] (echo-ish) '{normalized_stt}' ~ '{normalized_tts}'")
            return True

        similarity = Levenshtein.distance(normalized_tts, normalized_stt) / max(
            len(normalized_tts), 1
        )
        if similarity < 0.4:
            print(f"[STT] (fuzzy echo-ish, sim={similarity:.2f})")
            return True

        return False

    # ---------- main loop ----------

    def run(self):
        print("🎙️ STT Worker started.")
        try:
            self._ensure_recognizer()
        except Exception as e:
            print(f"[STTWorker] Initial model load failed: {e}")

        while self.running_flag():
            # 1) Wait until session is active (wake word triggered and inside window)
            self.session_active_event.wait()
            if not self.running_flag():
                break

            # 2) Wait until TTS/LLM signals it's time to listen (via READY_SIGNAL → beep)
            self.ready_to_listen_event.wait()
            if not self.running_flag():
                break
            if not self.session_active_event.is_set():
                # Session might have ended while we were waiting
                continue

            # Consume this ready signal: ONE user turn per READY_SIGNAL
            self.ready_to_listen_event.clear()

            # 3) Ensure TTS is not speaking (should already be clear after beep)
            while self.is_speaking_event.is_set() and self.running_flag():
                time.sleep(0.01)

            # 4) Single listen for this turn
            text = ""
            try:
                # Mark that STT is currently listening (for Vision echo-guard etc.)
                self.stt_listening_event.set()
                
                text = self.wake_app.listen_once(
                    self.vosk_recognizer,
                    self.wake_app.sr_recognizer,
                    SR_MIC_DEVICE_INDEX,
                )

            except sr.WaitTimeoutError:
                print("[STT] Listen timeout (no speech detected).")
                text = ""
                continue
            except Exception as e:
                print("🎙️ STT Worker Error during listen_once:", e)
                text = ""
                time.sleep(0.3)
            finally:
                # STT is no longer listening
                self.stt_listening_event.clear()

            text = (text or "").strip()
            print(f"[STT] Got text: {text!r}")
            if text:
                print(f"User: {text}")

            # 5) Handle empty / timeout: no LLM turn, just wait for next READY_SIGNAL
            if not text:
                # by pushing a special token to the LLM, but per your request we
                continue

            # 6) Language switch commands
            if self._handle_language_switch(text):
                # We already switched language and re-armed listening;
                # no LLM turn for this utterance.
                continue

            # 7) Optional logging of echo/noise (but do NOT re-listen in same turn)
            if self._is_self_echo(text):
                print(f"[STT] This looks like an echo of my own voice, "
                      f"but I'm treating it as a normal user turn to keep the "
                      f"turn-taking protocol simple.")

            # 8) Valid utterance → forward to LLM and finish this turn
            self.input_queue.put(text)

            # Back to top of loop:
            # - LLMWorker will answer
            # - TTSWorker + AudioPlaybackManager will speak
            # - AudioPlaybackManager will send READY_SIGNAL + beep
            # - only THEN will we listen again

            time.sleep(0.01)

        print(" STT Worker finished.")
