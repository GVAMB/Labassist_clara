#!/usr/bin/env python3
"""
main_app.py

Multimodal Assistant orchestrator:

- Wake word (Porcupine) via ClaraWakeApp
- Offline STT (Vosk) via ClaraWakeApp
- LabAssistRAG (OpenAI + LangChain RAG)
- TTS (Piper)
- Vision (InsightFace + embeddings DB + FaceRecognitionLogger)
- Emotion display (EmotionPlayer) on Raspberry Pi HDMI 1024x600

Key flow per session:
  Wake word -> greeting + beep
  STT -> LLM (RAG) -> TTS  [up to CONVERSATION_WINDOW_SEC]
  Vision -> "Vision System Report: ..." -> LLM -> TTS
"""

import time
import signal
import threading
from queue import Queue
from typing import Optional
from pathlib import Path

from config import (
    CONVERSATION_WINDOW_SEC,
    READY_SIGNAL,
    POISON_PILL,
    RunningFlag,
)
from STT_TTS.ClaraWakeApp import ClaraWakeApp
from vision.FaceRecognitionlogger import FaceRecognitionLogger
from STT_TTS.piper_tts import OfflineTTS
from LLM.LLM_engine import LabAssistRAG

from emotion_animator import (
    EmotionPlayer,
    _emotion_cmd_q,
    set_emotion_from_outside,
    BASE_PATH,
    FRAME_RATE,
    WINDOWED,
)

from stt_worker import STTWorker
from llm_worker import LLMWorker
from tts_worker import TTSWorker, AudioPlaybackManager
from vision_worker import VisionDynamicWorker


# =========================
# Wake Word Worker
# =========================

class WakeWordWorker(threading.Thread):
    def __init__(
        self,
        wake_app: ClaraWakeApp,
        wake_triggered_event: threading.Event,
        running_flag: RunningFlag,
        tts_playback_q: Queue,
        llm_output_q: Queue,
    ):
        super().__init__(daemon=True)
        self.wake_app = wake_app
        self.wake_triggered_event = wake_triggered_event
        self.running_flag = running_flag
        self.tts_playback_q = tts_playback_q
        self.llm_output_q = llm_output_q

    def run(self):
        print(" WakeWordWorker started. Waiting for wake word...")
        porcupine = None
        rec = None
        try:
            porcupine, wake_label = self.wake_app.init_porcupine()
            frame_length = porcupine.frame_length
            rec = self.wake_app.init_recorder(frame_length)
            print(f"[Wake] Say '{wake_label}' to start the assistant.")

            while self.running_flag():
                pcm = rec.read()
                if porcupine.process(pcm) >= 0:
                    print(" Wake word detected!")
                    self.wake_triggered_event.set()
                    time.sleep(2)  # small delay

                    greeting_text = "Hello, I am Clara. I’m really excited to see all of you. Let me quickly check who’s here before we start."
                    
                    self.llm_output_q.put(greeting_text)
                    self.llm_output_q.put(READY_SIGNAL)
                    set_emotion_from_outside("happy")

                    break
                time.sleep(0.005)

        except Exception as e:
            print(" WakeWordWorker Error:", e)
        finally:
            try:
                self.wake_app.clean_shutdown(rec, porcupine)
            except Exception:
                pass
            print(" WakeWordWorker finished.")


# =========================
# Assistant App Orchestrator
# =========================

class AssistantApp:
    def __init__(
        self,
        clara: ClaraWakeApp,
        rag_engine: LabAssistRAG,
        tts: OfflineTTS,
        vision: FaceRecognitionLogger,
        running_flag: RunningFlag | None = None,   
    ):
        # Use shared RunningFlag if provided (so eyes + assistant share it)
        self.running = running_flag or RunningFlag()
        self.input_q = Queue()
        self.output_q = Queue()
        self.enroll_q = Queue()
        self.tts_playback_q = Queue()
        self.is_speaking_event = threading.Event()
        self.ready_to_listen_event = threading.Event()
        self.is_enrolling_event = threading.Event()
        self.wake_triggered_event = threading.Event()
        self.session_active_event = threading.Event()
        self.stt_listening_event = threading.Event()

        self.clara = clara
        self.rag_engine = rag_engine
        self.tts = tts
        self.vision = vision

        self.wake_w: Optional[WakeWordWorker] = None

        # Workers
        self.stt_w: Optional[STTWorker] = None
        self.llm_w: Optional[LLMWorker] = None
        self.tts_w: Optional[TTSWorker] = None
        self.audio_playback_w: Optional[AudioPlaybackManager] = None
        self.vision_w: Optional[VisionDynamicWorker] = None

        #self.camera_lock = threading.Lock()

    def _start_background_workers_once(self):
        if self.stt_w is None:
            self.stt_w = STTWorker(
                wake_app=self.clara,
                input_queue=self.input_q,
                running_flag=self.running,
                is_speaking_event=self.is_speaking_event,
                ready_to_listen_event=self.ready_to_listen_event,
                session_active_event=self.session_active_event,
                stt_listening_event=self.stt_listening_event,
                llm_output_queue=self.output_q,
                audio_manager=self.audio_playback_w,
            )
            self.stt_w.start()

        if self.llm_w is None:
            self.llm_w = LLMWorker(
                rag_engine=self.rag_engine,
                input_queue=self.input_q,
                output_queue=self.output_q,
                running_flag=self.running,
                enroll_q=self.enroll_q,
                is_enrolling_event=self.is_enrolling_event,
                clara=self.clara,
                session_active_event=self.session_active_event,
                stt_listening_event=self.stt_listening_event, 
            )
            self.llm_w.start()


        if self.tts_w is None:
            # Important: route emotions to global emotion queue
            self.tts_w = TTSWorker(
                tts=self.tts,
                output_queue=self.output_q,
                running_flag=self.running,
                tts_playback_q=self.tts_playback_q,
                clara=self.clara,
                emotion_queue=_emotion_cmd_q,
                stt_listening_event=self.stt_listening_event,  
            )
            self.tts_w.start()


        if self.vision_w is None:
            self.vision_w = VisionDynamicWorker(
                vision_logger=self.vision,
                llm_input_queue=self.input_q,
                llm_output_queue=self.output_q,
                running_flag=self.running,
                enroll_queue=self.enroll_q,
                is_enrolling_event=self.is_enrolling_event,
                ready_to_listen_event=self.ready_to_listen_event,
                session_active_event=self.session_active_event,
                stt_listening_event=self.stt_listening_event,
                cam_index=0,
                width=640,
                height=480,
                det_width=320,
                sim_thresh=self.vision.sim_thresh if hasattr(self.vision, "sim_thresh") else 0.35,
                fps=self.vision.fps if hasattr(self.vision, "fps") else 2.0,
            )
            self.vision_w.start()
            # In AssistantApp class (Definition)


    def _clear_queues(self):
        """Clears all shared input/output queues to ensure a clean slate for the next session."""
        print(" Clearing queues of any leftover messages...")
        for q in [self.input_q, self.output_q, self.enroll_q, self.tts_playback_q]:
            while not q.empty():
                try:
                    # Clear item and mark as done
                    q.get_nowait()
                    q.task_done()
                except Exception:
                    # Should not happen, but prevents infinite loop if something goes wrong
                    break
    def run(self):
        """
        Orchestrator loop (NO pygame here).
        This is safe to run on a background thread.
        """
        print("🚀 Assistant starting (wake-word gated, repeated sessions)...")

        self.audio_playback_w = AudioPlaybackManager(
            tts=self.tts,
            audio_queue=self.tts_playback_q,
            running_flag=self.running,
            is_speaking_event=self.is_speaking_event,
            ready_to_listen_event=self.ready_to_listen_event,
        )
        self.audio_playback_w.start()

        self._start_background_workers_once()

        try:
            while self.running():
                # Reset session state
                self.session_active_event.clear()
                self.wake_triggered_event.clear()
                self.ready_to_listen_event.clear()

                print(" Waiting for wake word before (re)starting a session...")
                self.wake_w = WakeWordWorker(
                    wake_app=self.clara,
                    wake_triggered_event=self.wake_triggered_event,
                    running_flag=self.running,
                    tts_playback_q=self.tts_playback_q,
                    llm_output_q=self.output_q,
                )
                self.wake_w.start()

                # Wait for wake-word
                while self.running() and not self.wake_triggered_event.is_set():
                    time.sleep(0.1)

                if not self.running():
                    break

                print(
                    f" Wake word detected. Session is now active for "
                    f"{CONVERSATION_WINDOW_SEC:.0f} seconds."
                )
                self.session_active_event.set()
                session_start = time.time()

                while self.running() and self.session_active_event.is_set():
                    if time.time() - session_start > CONVERSATION_WINDOW_SEC:
                        print(" Session window elapsed. Blocking STT/Vision until next wake.")
                        self.session_active_event.clear()
                        self.ready_to_listen_event.clear()
                        try:
                            set_emotion_from_outside("sleep")
                        except Exception as e:
                            print(f"[Main] Failed to set bootup emotion: {e}")
                        break
                    time.sleep(0.1)

                # Put Clara to sleep visually.
                try:
                    set_emotion_from_outside("sleep")
                except Exception as e:
                    print(f"[AssistantApp] Failed to set sleep emotion: {e}")
                # --- START NEW SESSION CLEANUP ---
                self._clear_queues()

                if self.wake_w is not None:
                    self.wake_w.join(timeout=5)
                    self.wake_w = None


        except KeyboardInterrupt:
            print("\n KeyboardInterrupt in AssistantApp.run()")
            self.shutdown()
        else:
            self.shutdown()

    def shutdown(self):
        print(" Shutting down Assistant...")
        self.running.set_false()

        self.session_active_event.set()
        self.ready_to_listen_event.set()

        self.input_q.put(POISON_PILL)
        self.output_q.put(POISON_PILL)
        self.tts_playback_q.put(POISON_PILL)
        self.enroll_q.put(POISON_PILL)

        for w in (self.stt_w, self.llm_w, self.tts_w, self.audio_playback_w, self.vision_w, self.wake_w):
            if w is not None:
                w.join(timeout=5)

        try:
            self.tts.shutdown()
        except Exception as e:
            print("TTS shutdown error:", e)

        print("✅ All workers stopped.")


# =========================
# Main
# =========================

# =========================
# Main
# =========================

def _sigint(sig, frame):
    print("\nSIGINT received.")
    raise KeyboardInterrupt()


def assistant_thread_main(shared_running: RunningFlag):
    """
    Heavy initialization lives here (background thread):
    - ClaraWakeApp (Porcupine, Vosk)
    - LabAssistRAG
    - OfflineTTS
    - Vision logger
    - AssistantApp.run()
    """
    clara = ClaraWakeApp()
    rag_engine = LabAssistRAG()
    tts = OfflineTTS()
    vision = FaceRecognitionLogger(
        db_dir="vision/face_db",
        log_dir=Path("./recognition_logs"),
        keep_days=3,
    )

    app = AssistantApp(
        clara=clara,
        rag_engine=rag_engine,
        tts=tts,
        vision=vision,
        running_flag=shared_running,  
    )

    try:
        app.run()
    except KeyboardInterrupt:
        app.shutdown()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _sigint)

    #  Shared running flag for both assistant + emotion player
    shared_running = RunningFlag()

    # Create EmotionPlayer on MAIN THREAD (required by pygame/kmsdrm)
    eyes_player = EmotionPlayer(
        base_path=str(BASE_PATH),
        frame_rate=FRAME_RATE,
        windowed=WINDOWED,  # set WINDOWED=False in emotion_animator for robot mode
    )

    #  Immediately show bootup EMOTION before any heavy init (Clara, camera, etc.)
    try:
        set_emotion_from_outside("bootup")
    except Exception as e:
        print(f"[Main] Failed to set bootup emotion: {e}")

    # Start the heavy assistant initialization + orchestration in the background
    orchestrator_thread = threading.Thread(
        target=assistant_thread_main,
        args=(shared_running,),
        daemon=True,
    )
    orchestrator_thread.start()
    try:
        sleep.time(20)
        set_emotion_from_outside("happy")
    except Exception as e:
        print(f"[Main] Failed to set bootup emotion: {e}")


    try:
        print("🧠 Starting robot face loop on main thread...")
        # Robot face loop stays responsive the whole time
        eyes_player.run_robot_face(_emotion_cmd_q, shared_running)
    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt in main display loop")
        # Tell everyone to stop
        shared_running.set_false()
    finally:
        orchestrator_thread.join(timeout=5)
        print("👋 Main exited.")
