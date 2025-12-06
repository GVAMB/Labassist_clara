# main_wake_class.py — Clara wake-word assistant wrapped in a class
# Wake word (Porcupine) + offline STT (Vosk) + Piper TTS
# Bilingual EN<->DE, 15s conversation window after wake. Windows-safe mic handling.
# Matches main.py language switching & stop words.

import json
import signal
import sys
import time as _time
from pathlib import Path
from sys import platform

import pvporcupine
from pvrecorder import PvRecorder
import speech_recognition as sr
from vosk import Model, KaldiRecognizer

from STT_TTS.piper_tts import OfflineTTS

def find_pvrec_device_index():
    """
    Find PvRecorder device index for ReSpeaker by name.
    Prefer the *actual mic* over any 'Monitor of ...' loopback device.
    Returns an int index or -1 for system default.
    """
    devices = PvRecorder.get_available_devices()
    print("[PvRecorder] Available devices:")
    for i, name in enumerate(devices):
        print(f"  {i}: {name}")

    # 1) Prefer non-monitor ReSpeaker devices
    for i, name in enumerate(devices):
        if "Monitor of" in name:
            continue  # skip monitor/loopback devices
        if "ReSpeaker" in name or "Mic Array" in name or "USB Audio" in name:
            print(f"[PvRecorder] Using device {i}: {name}")
            return i

    # 2) As last fallback, allow a monitor ReSpeaker if nothing else was found
    for i, name in enumerate(devices):
        if "ReSpeaker" in name or "Mic Array" in name or "USB Audio" in name:
            print(f"[PvRecorder] Fallback using monitor device {i}: {name}")
            return i

    print("[PvRecorder] ReSpeaker not found, using system default (-1).")
    return -1
# ---------------------- CONFIG (aligned with main.py) ----------------------
ACCESS_KEY = "/NnA2SeSN4AMITZ5aAZZzBe9o4eSIUqS5scVQXE35ItyCsfm93pnDQ=="   # paste exact key; no extra spaces
SAMPLE_RATE = 16000
AMBIENT_CALIBRATION_SEC = 0.6
PHRASE_TIMEOUT_SEC = 25
LISTEN_TIMEOUT_SEC = 4

# Force the laptop microphone (Realtek array). If Windows reindexes devices, adjust this.
# For SR.Microphone (Vosk). PvRecorder uses device_index separately below.
SR_MIC_DEVICE_INDEX = None          # set to None for default
PVREC_DEVICE_INDEX = find_pvrec_device_index()         # -1 = system default for PvRecorder

FOLLOWUP_WINDOW_SEC = 200.0       # conversation window after wake
WAKE_COOLDOWN_SEC = 0.6          # avoid self-trigger from TTS

# Language registry (same paths/prompts as main.py)
LANGS = {
    "en": {
        "stt_model": Path("STT_TTS/models/vosk-model-small-en-us-0.15"),
        "prompts": {
            "ready":   "Hello, I'm CLARA, how can I help you?",
            "miss":    "Sorry, I didn't catch that.",
            "hello":   "Hello there!",
            "status":  "I'm Clara, doing great as always!",
            "goodbye": "Goodbye!",
        },
    },
    "de": {
        "stt_model": Path("STT_TTS/models/vosk-model-small-de-0.15"),
        "prompts": {
            "ready":   "Hallo, ich bin CLARA. Wie kann ich helfen?",
            "miss":    "Entschuldigung, das habe ich nicht verstanden.",
            "hello":   "Hallo! Wie kann ich dir helfen?",
            "status":  "Mir geht es gut, danke der Nachfrage.",
            "goodbye": "Auf Wiedersehen!",
        },
    },
}
DEFAULT_LANG = "en"

# Wake-word model choice
CUSTOM_PPN_PI = Path("STT_TTS/Hey-Clara_en_raspberry-pi_v3_0_0/hey_clara_raspberry-pi.ppn")
# ---------------------- PHRASES (match main.py’s strict style) ----------------------
def _norm_global(s: str) -> str:
    """Global version kept just in case, but class uses its own _norm method."""
    return " ".join((s or "").lower().strip().split())

# Exact flip phrases (normalized substring)
SWITCH_TO_DE = "can you speak german"
SWITCH_TO_EN = "sprechen sie english"

# Universal stop words (either language)
STOP_EN = {"stop", "exit", "quit", "goodbye", "good bye"}
STOP_DE = {
    "stopp", "stop", "stoppen", "halt", "beenden", "beende", "ende",
    "tschüss", "tschuss", "tschuess",
    "auf wiedersehen", "aufwiedersehen", "auf wieder sehn",
    "schließen", "schliessen", "programm beenden"
}
STOP_ALL = STOP_EN | STOP_DE


class ClaraWakeApp:
    def __init__(self):
        # CHANGED/NEW: run original sanity checks in __init__
        if not ACCESS_KEY.strip():
            raise SystemExit("ACCESS_KEY is empty. Paste the exact key from console.picovoice.ai.")
        for code in ("en", "de"):
            if not LANGS[code]["stt_model"].exists():
                raise SystemExit(f"Missing Vosk model at: {LANGS[code]['stt_model']}")

        self.current_lang = DEFAULT_LANG
        self.rec: PvRecorder | None = None
        self.porcupine: pvporcupine.Porcupine | None = None
        self.sr_recognizer = sr.Recognizer()
        self.tts= OfflineTTS()
    # ---------------------- Helpers as methods ----------------------

    def _norm(self, s: str) -> str:
        """Lowercase + collapse whitespace (same as before)."""
        # CHANGED: moved from global to method (added self)
        return " ".join((s or "").lower().strip().split())

    def init_porcupine(self):
        """Windows: built-in 'bumblebee'. Pi: prefer custom .ppn if present."""
        # CHANGED: now a method (added self), logic unchanged
        # if platform.startswith("win"):
        #     wake_word_label = "bumblebee"
        #     pp = pvporcupine.create(
        #         access_key=ACCESS_KEY,
        #         keywords=[wake_word_label],
        #         sensitivities=[0.65],
        #     )
        # else:
        if CUSTOM_PPN_PI.exists():
            wake_word_label = "hey clara"
            pp = pvporcupine.create(
                access_key=ACCESS_KEY,
                keyword_paths=[str(CUSTOM_PPN_PI)],
                sensitivities=[0.69],
            )
        else:
            wake_word_label = "bumblebee"
            pp = pvporcupine.create(
                access_key=ACCESS_KEY,
                keywords=[wake_word_label],
                sensitivities=[0.65],
            )
        return pp, wake_word_label

    def init_recorder(self, frame_length: int) -> PvRecorder:
        # CHANGED: method version uses global PVREC_DEVICE_INDEX (unchanged logic)
        rec = PvRecorder(device_index=PVREC_DEVICE_INDEX, frame_length=frame_length)
        rec.start()
        return rec

    def load_stt(self, lang_code: str):
        """Load (or reload) Vosk model + recognizer for a language."""
        # CHANGED: method version; logic unchanged
        model_dir = LANGS[lang_code]["stt_model"]
        model = Model(str(model_dir))
        recognizer = KaldiRecognizer(model, SAMPLE_RATE)
        return model, recognizer

    def listen_once(self, vosk_recognizer: KaldiRecognizer, r: sr.Recognizer, mic_index) -> str:
        r.dynamic_energy_threshold = True
        r.pause_threshold = 2.0
        r.non_speaking_duration = 0.80
        r.operation_timeout = LISTEN_TIMEOUT_SEC + PHRASE_TIMEOUT_SEC + 1

        try:
            with sr.Microphone(sample_rate=SAMPLE_RATE, device_index=mic_index) as source:
                r.adjust_for_ambient_noise(source, duration=AMBIENT_CALIBRATION_SEC)
                base = float(r.energy_threshold)
                r.energy_threshold = base * 1.1
                print(f"[MIC] Adjusted energy_threshold={r.energy_threshold:.1f}")

                print("Listening… (speak now)")
                audio = r.listen(
                    source,
                    timeout=LISTEN_TIMEOUT_SEC,
                    phrase_time_limit=PHRASE_TIMEOUT_SEC,
                )

        except sr.WaitTimeoutError:
            # Let STTWorker handle this as "no speech"
            raise
        except Exception as e:
            # Let STTWorker see the real error instead of pretending we got text
            print(f"[Clara listen_once] Mic / PyAudio error: {e}")
            raise

        data = audio.get_raw_data(convert_rate=SAMPLE_RATE, convert_width=2)
        if vosk_recognizer.AcceptWaveform(data):
            txt = json.loads(vosk_recognizer.Result()).get("text", "")
        else:
            txt = json.loads(vosk_recognizer.FinalResult()).get("text", "")
        return (txt or "").strip()



    # ---------------------- Language switch + command handling ----------------------

    def maybe_switch_language(self, t_norm: str, cur_lang: str) -> str | None:
        """Returns new lang code ('en'/'de') if a switch is requested; otherwise None."""
        # CHANGED: method version using global SWITCH_TO_*; logic unchanged
        if SWITCH_TO_DE in t_norm and cur_lang != "de":
            print("[LANG] Switching to DE (matched 'can you speak german')")
            return "de"
        if SWITCH_TO_EN in t_norm and cur_lang != "en":
            print("[LANG] Switching to EN (matched 'sprechen sie english')")
            return "en"
        return None

    def handle_command(self, text: str, lang: str) -> tuple[bool, str]:
        """
        Process a user utterance. Returns (continue_running, lang_code).
        continue_running=False signals the outer loop to exit the app.
        """
        # CHANGED: method version; calls self._norm / self.maybe_switch_language
        t = self._norm(text)
        prompts = LANGS[lang]["prompts"]

        if not t:
            self.tts.speak_text(prompts["miss"])
            return True, lang

        print(f"User: {text}")

        # Global stop (both languages)
        if any(kw in t for kw in STOP_ALL):
            self.tts.speak_text(prompts["goodbye"])
            return False, lang

        # Language switching
        new_lang = self.maybe_switch_language(t, lang)
        if new_lang:
            lang = new_lang
            self.tts.set_language(lang)  # swap Piper voice
            self.tts.speak_text(LANGS[lang]["prompts"]["ready"])
            return True, lang

        # Small talk
        if lang == "en":
            if "hello" in t:
                self.tts.speak_text(prompts["hello"])
            elif "how are you" in t:
                self.tts.speak_text(prompts["status"])
            else:
                self.tts.speak_text(f"You said: {text}")
        else:  # de
            if "hallo" in t:
                self.tts.speak_text(prompts["hello"])
            elif "wie geht" in t:
                self.tts.speak_text(prompts["status"])
            else:
                self.tts.speak_text(f"Du hast gesagt: {text}")

        return True, lang

    def converse_for_window(
        self,
        vosk_recognizer: KaldiRecognizer,
        r: sr.Recognizer,
        window_sec: float,
        lang: str
    ) -> tuple[bool, str]:
        """
        Keep doing STT -> reply cycles until the window expires.
        Returns (continue_app, current_lang).
        """
        # CHANGED: method version; uses self.listen_once / self.handle_command / self.load_stt / self._norm
        deadline = _time.time() + max(0.0, window_sec)
        print(f"[Conversation] Active for {window_sec:.0f}s — speak freely.")
        while _time.time() < deadline:
            try:
                text = self.listen_once(vosk_recognizer, r, SR_MIC_DEVICE_INDEX)
            except sr.WaitTimeoutError:
                continue
            except Exception:
                self.tts.speak_text("I encountered an error while listening.")
                continue

            # Handle command (may request stop or language switch)
            cont, lang = self.handle_command(text, lang)
            if not cont:
                return False, lang

            # If language changed, re-load Vosk model/recognizer to match it
            if SWITCH_TO_DE in self._norm(text) or SWITCH_TO_EN in self._norm(text):
                _, new_rec = self.load_stt(lang)
                vosk_recognizer = new_rec  # swap recognizer to the new language

            # Allow early end to window without exiting app
            if "go to sleep" in self._norm(text):
                self.tts.speak_text("Okay, going to sleep.")
                return True, lang

            # Optional: extend window after each exchange
            # deadline = _time.time() + max(0.0, window_sec)

        return True, lang  # window ended naturally

    def clean_shutdown(self, rec: PvRecorder | None, porcupine: pvporcupine.Porcupine | None):
        # CHANGED: method version; same cleanup logic
        for fn in (
            lambda: rec and rec.stop(),
            lambda: rec and rec.delete(),
            lambda: porcupine and porcupine.delete(),
        ):
            try:
                fn()
            except Exception:
                pass

    # ---------------------- MAIN LOOP as a method ----------------------

    def main(self):
        """Former top-level main() — now a method on the class."""
        porcupine, wake_word_label = self.init_porcupine()
        self.porcupine = porcupine

        # Start in English
        current_lang = DEFAULT_LANG
        self.current_lang = current_lang
        self.tts.set_language(current_lang)

        # Preload EN STT; the DE model will be loaded only if we switch
        model_en, recognizer = self.load_stt("en")
        r = self.sr_recognizer

        rec: PvRecorder | None = None
        self.rec = rec

        def _sigint(_sig, _frm):
            self.clean_shutdown(self.rec, self.porcupine)
            sys.exit(0)

        signal.signal(signal.SIGINT, _sigint)

        # Start Porcupine recorder
        rec = self.init_recorder(porcupine.frame_length)
        self.rec = rec
        self.tts.speak_text("Wake word is active.")
        print(f"[Wake] Say '{wake_word_label}' to start speaking.")

        last_wake = 0.0

        try:
            while True:
                pcm = rec.read()

                # Wake cooldown to avoid TTS self-trigger
                if (_time.time() - last_wake) < WAKE_COOLDOWN_SEC:
                    _time.sleep(0.005)
                    continue

                if porcupine.process(pcm) >= 0:
                    last_wake = _time.time()

                    # A) fully release hotword mic before any TTS/STT (Windows safety)
                    try:
                        rec.stop()
                        rec.delete()
                    except Exception:
                        pass

                    # Speak prompt in current language
                    self.tts.speak_text("Yes?" if current_lang == "en" else "Ja?")

                    # Ensure recognizer matches current language
                    _, new_recognizer = self.load_stt(current_lang)
                    recognizer = new_recognizer

                    continue_running, current_lang = self.converse_for_window(
                        recognizer, r, FOLLOWUP_WINDOW_SEC, current_lang
                    )
                    self.current_lang = current_lang
                    if not continue_running:
                        break  # user said stop/exit/goodbye

                    # D) Recreate recorder for next cycle
                    try:
                        rec = self.init_recorder(porcupine.frame_length)
                        self.rec = rec
                        # warm up a few frames
                        for _ in range(10):
                            rec.read()
                    except Exception:
                        self.tts.speak_text("Recorder error. Exiting.")
                        break

                    print(f"[Wake] Say '{wake_word_label}' to start speaking.")

                _time.sleep(0.005)

        except KeyboardInterrupt:
            pass
        finally:
            self.clean_shutdown(rec, porcupine)


if __name__ == "__main__":
    # NEW: create an object and call its main() instead of a free function
    app = ClaraWakeApp()
    app.main()
