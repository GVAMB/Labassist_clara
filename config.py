"""
config.py

Global constants and shared helpers for the Clara multimodal assistant.
Standard library only.
"""

import threading

# --------- Global Config Values ---------

DETECTION_INTERVAL_FRAMES = 6
FACE_THRESH = 0.35
VISION_FPS = 2.0  # Vision processes ~2 frames per second

# Conversation window length per wake (seconds)
CONVERSATION_WINDOW_SEC = 200.0

# Special tokens / signals
POISON_PILL = object()
READY_SIGNAL = "__BEEP_CUE_SIGNAL__"   # must never be normal LLM text
ENROLL_SIGNAL = "__START_ENROLL__"     # central constant for enroll trigger
EXIT_DIALOGUE = "__ABORT_DIALOGUE__"


# --------- Shared Running Flag + Beep ---------

class RunningFlag:
    """Thread-safe boolean flag used to stop all workers cleanly."""
    def __init__(self):
        self._running = True
        self._lock = threading.Lock()

    def __call__(self) -> bool:
        with self._lock:
            return self._running

    def set_false(self):
        with self._lock:
            self._running = False


