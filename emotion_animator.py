import os
import sys
import re
import random
import time
from pathlib import Path
from typing import Dict, List
from queue import Queue, Empty
import threading

import pygame  # import normally; we'll choose driver in __init__


# -------------------------------------------------------------------
# Global emotion queue + lock (used by other modules)
# -------------------------------------------------------------------
_emotion_cmd_q: "Queue[str]" = Queue()
_emotion_lock = threading.Lock()

# Default config
BASE_PATH = Path("robot_face_assets/emotions")
FRAME_RATE = 24
WINDOWED = False  # False = robot fullscreen, True = dev window


class EmotionPlayer:
    def __init__(self, base_path: str, frame_rate: int = 24, windowed: bool = True):
        self.base_path = base_path
        self.frame_rate = frame_rate

        if not os.path.isdir(self.base_path):
            print(f"[ERROR] Base path not found: {self.base_path}")
            sys.exit(1)

        # ---- Choose an SDL video driver that actually works ----
        # Try a few common ones on Raspberry Pi / Linux
        tried_drivers = []
        chosen_driver = None

        for driver_name in ["fbcon", "kmsdrm", "rpi", "x11", "wayland", "dummy"]:
            tried_drivers.append(driver_name)
            os.environ["SDL_VIDEODRIVER"] = driver_name
            try:
                pygame.display.init()
                chosen_driver = driver_name
                break
            except pygame.error:
                continue

        if chosen_driver is None:
            print(
                "[ERROR] Could not initialize any SDL video driver.\n"
                f"Tried: {', '.join(tried_drivers)}"
            )
            sys.exit(1)

        print(f"[EmotionPlayer] SDL video driver in use: {chosen_driver}")

        # If we ended up with dummy, we at least warn loudly
        if chosen_driver == "dummy":
            print(
                "[ERROR] SDL is using the 'dummy' video driver.\n"
                "No real display is available.\n"
                "Make sure you:\n"
                "  - Run this on the Pi's local console (HDMI), not only over SSH\n"
                "  - Have permissions to access the framebuffer/DRM device\n"
            )
            sys.exit(1)

        pygame.font.init()
        pygame.display.set_caption("Robot Face Emotions")

        if windowed:
            # Dev mode: small window (e.g. on your laptop)
            self.screen_size = (800, 480)
            self.screen = pygame.display.set_mode(self.screen_size)
        else:
            # ROBOT MODE: use 1024x600 borderless fullscreen
            self.screen_size = (1024, 600)
            flags = pygame.FULLSCREEN | pygame.NOFRAME
            self.screen = pygame.display.set_mode(self.screen_size, flags)

        print("[EmotionPlayer] Screen size:", self.screen.get_size())

        pygame.mouse.set_visible(False)
        self.clock = pygame.time.Clock()

        # font for overlay text (optional)
        self.font = pygame.font.SysFont(None, 26)

        # emotions & variants
        self.base_to_variants: Dict[str, List[str]] = {}
        self.base_order: List[str] = []

        self._scan_emotions()

        if not self.base_order:
            print(f"[ERROR] No emotion folders with PNG frames found under {self.base_path}")
            pygame.quit()
            sys.exit(1)

        # cache for pre-loaded frames
        self.cached_frames: Dict[str, List[pygame.Surface]] = {}

        # state
        self.current_base_idx = 0
        self.current_variant_idx = 0
        self.frames: List[pygame.Surface] = []
        self.frame_idx = 0
        self.paused = False

        # preload all frames at startup
        self._preload_all_frames()

    # ----------------- scanning & grouping ----------------- #

    def _scan_emotions(self) -> None:
        print(f"[INFO] Scanning emotion folders in {self.base_path}")

        for name in os.listdir(self.base_path):
            full = os.path.join(self.base_path, name)
            if not os.path.isdir(full):
                continue

            if not any(f.lower().endswith(".png") for f in os.listdir(full)):
                continue

            m = re.match(r"([A-Za-z_]+)", name)
            base = m.group(1).lower() if m else name.lower()

            self.base_to_variants.setdefault(base, []).append(name)

        for base in self.base_to_variants:
            self.base_to_variants[base].sort(key=str.lower)

        self.base_order = sorted(self.base_to_variants.keys(), key=str.lower)

        print("[INFO] Found base emotions and variants:")
        for base in self.base_order:
            print(f"  {base}: {self.base_to_variants[base]}")

    # ----------------- pre-loading frames ----------------- #

    def _load_frames_from_folder(self, folder_name: str) -> List[pygame.Surface]:
        folder = os.path.join(self.base_path, folder_name)
        files = sorted(
            (f for f in os.listdir(folder) if f.lower().endswith(".png")),
            key=lambda name: int(re.search(r"\d+", name).group()) if re.search(r"\d+", name) else 0,
        )
        if not files:
            raise ValueError(f"No PNG frames in {folder}")

        frames: List[pygame.Surface] = []
        for fname in files:
            path = os.path.join(folder, fname)
            try:
                img = pygame.image.load(path).convert_alpha()
            except Exception as e:
                print(f"[WARN] Failed to load {path}: {e}")
                continue

            img = pygame.transform.smoothscale(img, self.screen_size)
            frames.append(img)

        if not frames:
            raise ValueError(f"Failed to load any frames from {folder}")

        print(f"[INFO] Loaded {len(frames)} frames from {folder_name}")
        return frames

    def _preload_all_frames(self) -> None:
        print("[INFO] Preloading all emotion frames into memory...")
        for base in self.base_order:
            for variant in self.base_to_variants[base]:
                if variant not in self.cached_frames:
                    self.cached_frames[variant] = self._load_frames_from_folder(variant)

    # ----------------- emotion selection ----------------- #

    def set_emotion(self, emotion_name: str):
        emotion_name = (emotion_name or "").strip().lower()
        if not emotion_name:
            return

        if emotion_name in self.base_to_variants:
            print(f"[INFO] Setting emotion to: {emotion_name}")
            self.current_base_idx = self.base_order.index(emotion_name)
            self.current_variant_idx = 0
            self._load_current_variant_frames()
        else:
            print(f"[WARN] Emotion '{emotion_name}' not found. Defaulting to 'neutral'.")
            if "neutral" in self.base_to_variants:
                self.set_emotion("neutral")

    def play_base(self, base_name: str, random_variant: bool = True) -> None:
        base = base_name.lower()
        if base not in self.base_to_variants:
            raise ValueError(f"Unknown emotion base: {base_name}")

        self.current_base_idx = self.base_order.index(base)
        variants = self.base_to_variants[base]

        if random_variant and len(variants) > 1:
            chosen = random.choice(variants)
            self.current_variant_idx = variants.index(chosen)
        else:
            self.current_variant_idx = 0

        self._load_current_variant_frames()

    def _load_current_variant_frames(self) -> None:
        base = self.base_order[self.current_base_idx]
        variants = self.base_to_variants[base]
        variant_folder = variants[self.current_variant_idx]

        print(f"[INFO] Using base='{base}', variant folder='{variant_folder}'")
        self.frames = self.cached_frames[variant_folder]
        self.frame_idx = 0

    # ----------------- overlay text (optional) ----------------- #

    def _draw_overlay(self) -> None:
        base = self.base_order[self.current_base_idx]
        variant_list = self.base_to_variants[base]
        text = (
            f"Emotion: {base}  |  Variant: {variant_list[self.current_variant_idx]}  "
            f"({self.current_variant_idx + 1}/{len(variant_list)})"
        )
        help_text = "SPACE pause  ESC quit"

        for dx, dy in ((1, 1), (0, 0)):
            color_main = (0, 0, 0) if (dx, dy) == (1, 1) else (255, 255, 255)
            txt_surface = self.font.render(text, True, color_main)
            self.screen.blit(txt_surface, (10 + dx, 10 + dy))

            color_help = (0, 0, 0) if (dx, dy) == (1, 1) else (200, 200, 200)
            help_surface = self.font.render(help_text, True, color_help)
            self.screen.blit(help_surface, (10 + dx, 35 + dy))

    # ----------------- robot mode loop (MAIN THREAD) ----------------- #

    def run_robot_face(self, emotion_queue: "Queue[str]", running_flag) -> None:
        if "neutral" in self.base_to_variants:
            self.play_base("neutral", random_variant=False)
        else:
            first_base = self.base_order[0]
            self.play_base(first_base, random_variant=False)

        while running_flag():
            try:
                while True:
                    new_emotion = emotion_queue.get_nowait()
                    new_emotion = (new_emotion or "").strip().lower()
                    if new_emotion and new_emotion in self.base_to_variants:
                        try:
                            self.play_base(new_emotion, random_variant=True)
                        except Exception as e:
                            print(f"[Eyes] Failed to play emotion '{new_emotion}': {e}")
            except Empty:
                pass

            if not self.paused and self.frames:
                self.frame_idx = (self.frame_idx + 1) % len(self.frames)

            if self.frames:
                frame = self.frames[self.frame_idx]
                self.screen.blit(frame, (0, 0))

            # self._draw_overlay()  # Uncomment if you want overlay text

            pygame.display.flip()
            self.clock.tick(self.frame_rate)

        pygame.quit()


# -------------------------------------------------------------------
# Helper for other modules
# -------------------------------------------------------------------
def set_emotion_from_outside(emotion: str):
    emotion = (emotion or "").strip().lower()
    if not emotion:
        return

    with _emotion_lock:
        _emotion_cmd_q.put_nowait(emotion)


# Optional standalone test
if __name__ == "__main__":
    player = EmotionPlayer(str(BASE_PATH), FRAME_RATE, WINDOWED)

    running = True

    def running_flag():
        return running

    def random_emotions():
        bases = player.base_order
        while running:
            time.sleep(3)
            emo = random.choice(bases)
            print(f"[TEST] Emotion -> {emo}")
            set_emotion_from_outside(emo)

    t = threading.Thread(target=random_emotions, daemon=True)
    t.start()

    try:
        player.run_robot_face(_emotion_cmd_q, running_flag)
    except KeyboardInterrupt:
        running = False
        print("[TEST] Stopping...")
