import pygame
import os
import time

# Define the path where the frames are stored
BASE_PATH = r"F:\studies\sem7\industrial_proj\final_codes2_per\robot_face_assets\emotions"

# Frame rate for animation (frames per second)
FRAME_RATE = 24

# Function to load frames from a folder
def load_frames_from_folder(folder_name):
    folder_path = os.path.join(BASE_PATH, folder_name)
    frames = []

    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print(f"[ERROR] Folder not found: {folder_path}")
        return frames

    # Load each PNG file from the folder
    for file in os.listdir(folder_path):
        if file.lower().endswith(".png"):
            frame_path = os.path.join(folder_path, file)
            try:
                # Load the image and add it to the frames list
                img = pygame.image.load(frame_path).convert_alpha()
                frames.append(img)
            except pygame.error as e:
                print(f"[ERROR] Failed to load image {frame_path}: {e}")

    return frames

# Initialize Pygame
pygame.init()

# Set up the display window
info = pygame.display.Info()
screen_size = (int(info.current_w * 0.8), int(info.current_h * 0.8))  # Use 80% of the screen size
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption("Emotion Animation Test")

# Load the frames for a specific emotion (you can change the emotion folder here)
emotion_name = "happy"  # Change this to any emotion folder in your BASE_PATH
frames = load_frames_from_folder(emotion_name)

# Check if frames were loaded successfully
if not frames:
    #print("[ERROR] No frames loaded. Please check your folder or image format.")
    pygame.quit()
    exit()

# Clock to control the frame rate (FPS)
clock = pygame.time.Clock()

# Main loop for testing the animation
running = True
frame_idx = 0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen (fill with black)
    screen.fill((0, 0, 0))

    # Draw the current frame on the screen
    current_frame = frames[frame_idx]
    screen.blit(current_frame, (0, 0))  # Draw the frame at (0,0) on the screen

    # Update the display with the new frame
    pygame.display.flip()

    # Increment the frame index to move to the next frame
    frame_idx = (frame_idx + 1) % len(frames)

    # Control the animation speed (FPS)
    clock.tick(FRAME_RATE)

    # Pause for a short time to avoid maxing out CPU usage
    time.sleep(0.01)

# Clean up and quit Pygame
pygame.quit()
