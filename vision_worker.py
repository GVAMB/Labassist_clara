"""
vision_worker.py

Contains VisionDynamicWorker:
- Owns the camera + InsightFace
- Uses FaceRecognitionLogger for DB + recognition_core
- Sends 'Vision System Report: ...' messages to LLM
- Handles voice-driven enrollment (name + role) and adds embeddings to DB
- Respects session_active_event and STT echo-guard to avoid self-talk
"""

import time
import threading
from queue import Queue, Empty
from typing import Optional
from pathlib import Path
import json
import numpy as np
from numpy.linalg import norm
import cv2
from insightface.app import FaceAnalysis
from picamera2 import Picamera2  

from config import (
    FACE_THRESH,
    VISION_FPS,
    POISON_PILL,
    READY_SIGNAL,
    ENROLL_SIGNAL,
    EXIT_DIALOGUE,
    RunningFlag,
)
from vision.FaceRecognitionlogger import FaceRecognitionLogger


class VisionDynamicWorker(threading.Thread):
    """
    Single worker that:
      - Owns the camera + InsightFace
      - Uses FaceRecognitionLogger for DB + recognition_core
      - Sends 'Vision System Report: ...' messages to LLM
      - Handles voice-driven enrollment (name + role) and adds embeddings to DB

    Only runs recognition while session_active_event is set.
    Uses stt_listening_event + ready_to_listen_event to avoid echo/self-talk.
    """

    def __init__(
        self,
        vision_logger: FaceRecognitionLogger,
        llm_input_queue: Queue,
        llm_output_queue: Queue,
        running_flag: RunningFlag,
        enroll_queue: Queue,
        is_enrolling_event: threading.Event,
        ready_to_listen_event: threading.Event,
        session_active_event: threading.Event,
        stt_listening_event: threading.Event,
        cam_index: int = 0,         # kept for API compatibility, unused with Picamera2
        width: int = 640,
        height: int = 480,
        det_width: int = 320,
        sim_thresh: float = FACE_THRESH,
        fps: float = VISION_FPS,
    ):
        super().__init__(daemon=True)

        # Queues & flags
        self.vision = vision_logger
        self.llm_input_q = llm_input_queue
        self.llm_output_q = llm_output_queue
        self.running_flag = running_flag
        self.enroll_q = enroll_queue
        self.is_enrolling_event = is_enrolling_event
        self.ready_to_listen_event = ready_to_listen_event
        self.session_active_event = session_active_event
        self.stt_listening_event = stt_listening_event

        # Basic config
        self.cam_index = cam_index          # not used with Picamera2, but kept for compatibility
        self.width = width
        self.height = height
        self.det_width = det_width
        self.sim_thresh = sim_thresh
        self.fps = max(0.1, fps)

        # Debounce / state
        self.last_state = set()
        self.pending_state = None
        self.pending_count = 0
        self.stable_threshold = 3
        self._first_run_after_active = False
        self.last_total_faces = 0

        # DB
        self.vision.ensure_db()
        self.E, self.N, self.M = self.vision.load_embeddings()

        # InsightFace – CPU only on Pi
        self.app = FaceAnalysis(providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(self.det_width, self.det_width))

        # --------- Picamera2 setup (robust) ---------
        self.picam = None
        self.camera_ok = False
        #self.camera_lock = threading.Lock()

        try:
            infos = Picamera2.global_camera_info()
            print(f"[Vision] Picamera2.global_camera_info() -> {infos}")
        except Exception as e:
            print(f"[Vision] ERROR calling Picamera2.global_camera_info(): {e}")
            infos = []

        if not infos:
            print("[Vision] ERROR: No cameras reported by Picamera2. Vision will idle.")
        else:
            try:
                self.picam = Picamera2()
                main_cfg = {
                    "size": (self.width, self.height),
                    "format": "YUV420",   # same as your working logger
                }
                cfg = self.picam.create_video_configuration(main=main_cfg, buffer_count=4)
                self.picam.configure(cfg)
                self.picam.start()
                time.sleep(1.0)  # AE/AWB settle
                self.camera_ok = True
                print("[Vision] Picamera2 initialized and started.")
            except Exception as e:
                print(f"[Vision] ERROR: Failed to create/start Picamera2 instance: {e}")
                self.camera_ok = False

    # ----------------- helpers -----------------

    def _make_state_set(self, known_list):
        return {f"{p['name']}_{p.get('role', '')}" for p in known_list}

    def _append_person_embedding(self, name: str, role_str: str, emb: np.ndarray):
        E, N, M = self.vision.load_embeddings()

        emb = emb.astype(np.float32)
        emb = emb / (norm(emb) + 1e-9)

        r = (role_str or "").strip().lower()
        role_db = "student" if "student" in r or "study" in r else "staff"

        names_list = list(N)
        if names_list and name in names_list:
            idx = names_list.index(name)
            E[idx] = emb
            print(f"[DB] Overwriting embedding for {name} ({role_db}).")
        else:
            E = np.vstack([E, emb]) if E.size else emb[None, :]
            N = np.append(N, name)
            print(f"[DB] Adding new person {name} ({role_db}).")

        sig = f"{np.random.randint(0,0xFFFF):04X}-{np.random.randint(0,0xFFFF):04X}"
        M[name] = {"role": role_db, "signature": sig}

        np.savez_compressed(self.vision.EMB_FILE, embeddings=E.astype(np.float32), names=N)
        with open(self.vision.META_FILE, "w", encoding="utf-8") as f:
            json.dump(M, f, indent=2, ensure_ascii=False)

        self.E, self.N, self.M = E, N, M
        print(f"[DB] Stored {name} ({role_db}) Sig:{sig}")

    # --------- capture from Picamera2, not VideoCapture ---------
    def _capture_frame(self):
        """Thread-safe capture from Picamera2. Returns BGR frame or None."""
        if not self.camera_ok or self.picam is None:
            return None

        #with self.camera_lock:
        try:
            yuv = self.picam.capture_array("main")
        except Exception as e:
            print(f"[Vision] WARN: Picamera2 capture_array failed: {e}")
            return None

        # YUV420 → BGR (same as in your logger)
        try:
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        except Exception as e:
            print(f"[Vision] WARN: cvtColor YUV->BGR failed: {e}")
            return None

        return frame

    def _capture_enrollment_embedding(self, shots=20, timeout_sec=30):
        print("\n[ENROLL] Starting capture from camera.")
        print(f"[ENROLL] Collecting {shots} samples (timeout {timeout_sec}s)...")

        if not self.camera_ok:
            print("[ENROLL] Camera not available, cannot capture enrollment.")
            return None

        samples = []
        start = time.time()

        while len(samples) < shots and (time.time() - start) < timeout_sec and self.running_flag():
            frame = self._capture_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            faces = self.app.get(frame)
            if not faces:
                continue

            f = max(
                faces,
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
            )
            emb = getattr(f, "normed_embedding", None)
            if emb is None or emb.size == 0:
                continue

            samples.append(emb.astype(np.float32))
            print(f"[ENROLL] Collected {len(samples)}/{shots}", end="\r", flush=True)

        print()
        if len(samples) < 5:
            print("[ENROLL] Not enough samples.")
            return None

        emb_mean = np.mean(np.vstack(samples), axis=0)
        emb_mean = emb_mean / (norm(emb_mean) + 1e-9)
        print(f"[ENROLL] Collected {len(samples)} samples.")
        return emb_mean

    def _wait_for_user_response(self, timeout=10):
        # trigger normal STT flow via READY_SIGNAL -> beep -> listen
        self.llm_output_q.put(READY_SIGNAL)
        try:
            resp = self.llm_input_q.get(timeout=timeout)
            if resp is POISON_PILL:
                return None
            text = str(resp).strip()
            self.llm_input_q.task_done()
        except Empty:
            self.llm_output_q.put("I didn't hear you. Let's try again later.")
            return None

        low = text.lower()
        if any(w in low for w in ["stop", "cancel", "never mind", "abort"]):
            self.llm_output_q.put("Enrollment cancelled. What else can I help you with?")
            return EXIT_DIALOGUE

        return text

    def _handle_enrollment(self):
        print(" [VisionDynamicWorker] Enrollment flow started.")

        if not self.camera_ok:
            self.llm_output_q.put("My camera is currently unavailable, I can't enroll new faces right now.")
            self.is_enrolling_event.clear()
            self.llm_output_q.put(READY_SIGNAL)
            return

        name = None
        while self.running_flag() and not name:
            self.llm_output_q.put("What is your name?")
            name = self._wait_for_user_response()
            if name == EXIT_DIALOGUE:
                self.is_enrolling_event.clear()
                self.llm_output_q.put(READY_SIGNAL)
                return
            if not name:
                self.is_enrolling_event.clear()
                self.llm_output_q.put(READY_SIGNAL)
                return

        role = None
        while self.running_flag() and not role:
            self.llm_output_q.put(f"Thank you, {name}. Are you a student or a professor?")
            time.sleep(0.05)
            role = self._wait_for_user_response()
            if role == EXIT_DIALOGUE:
                self.is_enrolling_event.clear()
                self.llm_output_q.put(READY_SIGNAL)
                return
            if not role:
                self.llm_output_q.put("I didn't catch your role. Let's stop enrollment for now.")
                self.is_enrolling_event.clear()
                self.llm_output_q.put(READY_SIGNAL)
                return

        role_str = role.lower()
        role_norm = "student" if "student" in role_str or "study" in role_str else "professor"

        self.llm_output_q.put(
            f"Okay {name}, please look at the camera while I register your face."
        )
        emb = self._capture_enrollment_embedding()
        if emb is None:
            self.llm_output_q.put("I couldn't get a good capture. Let's try again later.")
            self.is_enrolling_event.clear()
            self.llm_output_q.put(READY_SIGNAL)
            return

        self._append_person_embedding(name, role_norm, emb)
        self.llm_output_q.put(f"Enrollment successful for {name} ({role_norm}).")
        self.is_enrolling_event.clear()
        self.llm_output_q.put(READY_SIGNAL)

    # ----------------- main loop -----------------

    def run(self):
        print(" VisionDynamicWorker started.")
        frame_interval = 1.0 / self.fps

        # If camera failed to init, just idle instead of crashing the whole app
        if not self.camera_ok:
            print("[Vision] Camera not available – VisionDynamicWorker will idle.")
            while self.running_flag():
                time.sleep(1.0)
            print(" VisionDynamicWorker finished (no camera).")
            return

        try:
            while self.running_flag():

                # Session gating
                if not self.session_active_event.is_set():
                    # We still tick the camera to keep it alive, but no recognition/reporting
                    _ = self._capture_frame()
                    self._first_run_after_active = True
                    self.last_total_faces = 0
                    time.sleep(0.1)
                    continue

                start_time = time.time()

                # 1) Check enrollment trigger
                try:
                    req = self.enroll_q.get_nowait()
                    if req == ENROLL_SIGNAL:
                        self.enroll_q.task_done()
                        self._handle_enrollment()
                except Empty:
                    pass

                # 2) Grab frame
                frame = self._capture_frame()
                if frame is None:
                    print("[Vision] WARN: Failed to read frame from Picamera2")
                    time.sleep(0.05)
                    continue

                # 3) Run recognition
                faces_out, counts = self.vision.recognize_frame(
                    self.app, frame, self.E, self.N, self.M, self.sim_thresh,
                )

                known_faces = [f for f in faces_out if f["name"] != "UNKNOWN"]
                num_faces = counts.get("total", len(faces_out))
                num_known = len(known_faces)
                num_unknown = counts.get("unknown", max(0, num_faces - num_known))

                # ---------- INITIAL COUNT REPORT (STRUCTURED) ----------
                if self._first_run_after_active:
                    if self.session_active_event.is_set():
                        if num_faces == 0:
                            # Nobody visible
                            report_msg = (
                                "Vision System Report: INITIAL_COUNT "
                                "total_people=0; known_faces=none; unidentified_people=0."
                            )
                        else:
                            # Build structured known_faces field
                            if num_known > 0:
                                known_descriptions = []
                                for f in known_faces:
                                    role = f.get("role") or ""
                                    if role:
                                        known_descriptions.append(f"{f['name']} ({role})")
                                    else:
                                        known_descriptions.append(f"{f['name']}")
                                known_str = ", ".join(known_descriptions)
                            else:
                                known_str = "none"

                            report_msg = (
                                "Vision System Report: INITIAL_COUNT "
                                f"total_people={num_faces}; "
                                f"known_faces={known_str}; "
                                f"unidentified_people={num_unknown}."
                            )

                        print(" -> Initial Vision Report:", report_msg)
                        self.llm_input_q.put(report_msg)
                        self.last_state = self._make_state_set(known_faces)
                        self.pending_state = frozenset(self.last_state)
                        self.pending_count = 0
                        self.last_total_faces = num_faces
                        self._first_run_after_active = False
                    else:
                        print("[Vision] Skipping initial report (session inactive).")
                        self._first_run_after_active = False

                # Fixed FPS
                elapsed = time.time() - start_time
                remaining = frame_interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        except Exception as e:
            print(" VisionDynamicWorker Error:", e)
        finally:
            try:
                if self.picam is not None:
                    self.picam.stop()
            except Exception:
                pass
            print(" VisionDynamicWorker finished.")

