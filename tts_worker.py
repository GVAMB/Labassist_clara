# tts_worker.py

import time
import threading
from queue import Queue
from typing import Tuple, Any

from config import (
    POISON_PILL,
    READY_SIGNAL,
    RunningFlag,
)
from STT_TTS.piper_tts import OfflineTTS
from STT_TTS.ClaraWakeApp import ClaraWakeApp 
import numpy as np
import sounddevice as sd  # Required here for playback in AudioPlaybackManager

class TTSWorker(threading.Thread):
    """
    TTSWorker is the SYNTHESIZER/TEXT DISPATCHER.
    It takes text from the LLM, synthesizes it into raw audio data (non-blocking), 
    and sends the raw data to the AudioPlaybackManager queue.
    """

    def __init__(
        self,
        tts: OfflineTTS,
        output_queue: Queue,
        running_flag: RunningFlag,
        tts_playback_q: Queue,
        clara: ClaraWakeApp,
        emotion_queue,
        stt_listening_event: threading.Event,
    ):
        super().__init__(daemon=True)
        self.tts = tts
        self.output_queue = output_queue
        self.running_flag = running_flag
        self.tts_playback_q = tts_playback_q
        self.clara = clara
        self.emotion_queue = emotion_queue
        self.stt_listening_event = stt_listening_event

    def run(self):
        print(" TTS Worker started (Synthesizer/Text dispatcher).")
        while self.running_flag():
            item = self.output_queue.get()
            try:
                # --- Shutdown handling ---
                if item is POISON_PILL:
                    self.tts_playback_q.put(POISON_PILL)
                    self.output_queue.task_done()
                    break

                # --- Just forward READY_SIGNAL to AudioPlaybackManager ---
                if item == READY_SIGNAL:
                    print(" READY_SIGNAL received, forwarding for post-playback cue.")
                    self.tts_playback_q.put(READY_SIGNAL)
                    self.output_queue.task_done()
                    continue
                # --- Text / emotion extraction ---
                emotion = "neutral"
                text = ""

                if isinstance(item, dict) and item.get("type") == "speech":
                    text = item.get("text", "")
                    emotion = item.get("emotion", "neutral")
                else:
                    text = str(item)

                # Pass emotion to eye animator (non-blocking)
                try:
                    self.emotion_queue.put_nowait(emotion)
                except Exception:
                    pass

                if not text:
                    # Nothing to say, just skip
                    self.output_queue.task_done()
                    continue

                # 1. Sync TTS voice (Must happen BEFORE synthesis)
                current_lang = getattr(self.clara, "current_lang", "en")
                if self.tts.get_current_language() != current_lang:
                    self.tts.set_language(current_lang)

                # 2. Synthesize audio (NON-BLOCKING) and apply emotion transforms
                print(f" TTS (Synthesizing): {text}")
                start = time.time()
                audio_data, sr = self.tts.synthesize_with_emotion_raw(text, emotion)
                synth_time = time.time() - start

                approx_len = 0.0
                if audio_data.size> 0 and sr > 0:
                     approx_len = audio_data.shape[0] / sr
                print(f"[TTS] Synthesis took {synth_time:.2f}s for ~{approx_len:.2f}s of audio")      

                if audio_data.size > 0:
                    # 3. Queue the raw data and metadata for playback
                    self.tts_playback_q.put((audio_data, sr, text))

                self.output_queue.task_done()

            except Exception as e:
                print(" TTS Worker Error:", e)
                self.output_queue.task_done()
                time.sleep(0.5)

        print(" TTS Worker finished.")


class AudioPlaybackManager(threading.Thread):
    """
    AudioPlaybackManager is the PLAYER and THREAD MANAGER.
    It takes raw audio data, plays it using a blocking call (sd.play), and controls 
    the is_speaking_event to prevent microphone interference/overlap.
    """

    def __init__(
        self,
        tts: OfflineTTS,
        audio_queue: Queue,
        running_flag: RunningFlag,
        is_speaking_event: threading.Event,
        ready_to_listen_event: threading.Event,
    ):
        super().__init__(daemon=True)
        self.tts = tts
        self.audio_queue = audio_queue
        self.running_flag = running_flag
        self.is_speaking_event = is_speaking_event
        self.ready_to_listen_event = ready_to_listen_event
        self.last_spoken_text = ""

    def shutdown(self):
        """Alias for compatibility with code that calls tts.shutdown()."""
        try:
            self.close_audio()
        except Exception:
            pass

    def run(self):
        print("🎧 Audio Playback Manager started.")
        while self.running_flag():
            item = self.audio_queue.get()
            try:
                if item is POISON_PILL:
                    break

                if item == READY_SIGNAL:
                    # 1. Clear the speaking event. This tells STT it can start listening.
                    self.is_speaking_event.clear()
                    # 2. Beep and cue STT
                    self.ready_to_listen_event.set()
                    continue 
                else:
                    # item is (audio_data, sr, text) from the TTSWorker
                    if isinstance(item, (tuple, list)) and len(item) == 3:
                        audio_data, sr, text = item 
                    else:
                        print(f" Audio Playback Error: Received unexpected item type/length: {item!r}. Skipping playback.")
                        continue

                    # Store the text for echo-guarding
                    self.last_spoken_text = text

                    # Set the speaking event (prevents STT listening during playback)
                    self.is_speaking_event.set()

                    print(f" Playing back chunk: {text}")

                    # Play on the selected OUTPUT_DEVICE_INDEX
                    sd.play(audio_data, samplerate=sr, blocking=True)

            except Exception as e:
                print(" Audio Playback Error:", e)
                # On error, clear the speaking event to prevent system lockup
                self.is_speaking_event.clear()
                time.sleep(0.2)
            finally:
                self.audio_queue.task_done()

        print(" Audio Playback Manager finished.")
        self.tts.close_audio()  # Clean up sounddevice
