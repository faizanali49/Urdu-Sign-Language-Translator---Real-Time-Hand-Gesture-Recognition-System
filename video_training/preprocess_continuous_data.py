# preprocess_continuous_data.py
import cv2
import mediapipe as mp
import numpy as np
import os
from utils import normalize_landmarks # Import the normalization function

# --- Configuration ---
DATASET_DIR = "continuous_dataset"
VIDEO_DIR = os.path.join(DATASET_DIR, "videos")
LANDMARKS_DIR = os.path.join(DATASET_DIR, "landmarks_normalized") # Save normalized data here
MIN_FRAMES = 10 # Minimum number of frames with detected hands to save a sequence

# --- Initialization ---
os.makedirs(LANDMARKS_DIR, exist_ok=True)
print(f"Normalized landmarks will be saved to: {os.path.abspath(LANDMARKS_DIR)}")

mp_hands = mp.solutions.hands
# Process as video stream, detect one hand
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)

print("Starting preprocessing...")
total_videos_processed = 0
total_sequences_saved = 0
total_sequences_skipped = 0

# --- Main Loop ---
gesture_folders = [d for d in os.listdir(VIDEO_DIR) if os.path.isdir(os.path.join(VIDEO_DIR, d))]
if not gesture_folders:
    print(f"Error: No gesture folders found in {VIDEO_DIR}")
    exit()

for gesture_name in gesture_folders:
    print(f"\nProcessing gesture: {gesture_name}")
    gesture_video_path = os.path.join(VIDEO_DIR, gesture_name)
    gesture_landmark_path = os.path.join(LANDMARKS_DIR, gesture_name)
    os.makedirs(gesture_landmark_path, exist_ok=True)

    video_files = [f for f in os.listdir(gesture_video_path) if f.lower().endswith((".avi", ".mp4", ".mov"))]
    if not video_files:
        print("  No video files found.")
        continue

    for video_file in video_files:
        video_path = os.path.join(gesture_video_path, video_file)
        print(f"  Processing video: {video_file}")
        total_videos_processed += 1

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"    Error: Could not open video file {video_path}")
            total_sequences_skipped += 1
            continue

        raw_landmark_sequence = []
        frame_count = 0
        hand_detected_count = 0

        # --- Extract Raw Landmarks ---
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # End of video

            frame_count += 1
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False # Optimize

            # Process the image and find hands
            result = hands.process(rgb_frame)

            # Extract landmarks if hand detected, otherwise append placeholder (zeros)
            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0] # Assuming max_hands=1
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32).flatten()
                raw_landmark_sequence.append(landmarks)
                hand_detected_count += 1
            else:
                # Append placeholder for frames with no hand detected
                raw_landmark_sequence.append(np.zeros(63, dtype=np.float32))

        cap.release()

        # --- Normalize and Save Sequence ---
        if not raw_landmark_sequence:
             print("    Warning: No frames processed from video.")
             total_sequences_skipped += 1
             continue

        if hand_detected_count < MIN_FRAMES:
            print(f"    Skipping: Not enough hand frames detected ({hand_detected_count}/{frame_count}). Need at least {MIN_FRAMES}.")
            total_sequences_skipped += 1
            continue

        # Convert list to NumPy array
        raw_landmark_sequence_np = np.array(raw_landmark_sequence)

        # ** Apply Normalization **
        normalized_sequence = normalize_landmarks(raw_landmark_sequence_np)

        if normalized_sequence is not None:
            # Save the normalized sequence
            output_filename_base = os.path.splitext(video_file)[0] # Remove original extension
            output_filename = os.path.join(gesture_landmark_path, f"{output_filename_base}.npy")
            try:
                np.save(output_filename, normalized_sequence)
                print(f"    ✅ Saved normalized landmarks ({normalized_sequence.shape}) to {output_filename}")
                total_sequences_saved += 1
            except Exception as e:
                print(f"    Error saving normalized landmarks for {video_file}: {e}")
                total_sequences_skipped += 1
        else:
            print(f"    Skipping: Normalization failed for {video_file}.")
            total_sequences_skipped += 1


# --- Cleanup ---
hands.close()
print("\nPreprocessing complete!")
print(f"Total videos processed: {total_videos_processed}")
print(f"Total sequences saved: {total_sequences_saved}")
print(f"Total sequences skipped: {total_sequences_skipped}")